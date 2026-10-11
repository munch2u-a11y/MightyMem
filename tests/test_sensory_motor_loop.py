"""Unit tests for the 3-stage Small -> Big -> Small Sensory-Motor Turn Loop."""
from __future__ import annotations

import pytest
import torch
from pathlib import Path

from habitus_ai.sensory_motor_loop import (
    SensoryEncoder,
    MotorDecoder,
    SensoryMotorTurnLoop,
    ActionPrimitive,
)
from context_loom.engine import ContextLoom
from context_loom.runtime import BoundedFileRuntime


def test_sensory_encoder_file_shapes(tmp_path: Path) -> None:
    encoder = SensoryEncoder(vector_dim=256)

    py_file = tmp_path / "script.py"
    py_file.write_text("import os\nprint('hello')\n")

    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 header bytes\x00\x01\x02")

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Just plain text notes.")

    s_py = encoder.encode_file(py_file)
    s_pdf = encoder.encode_file(pdf_file)
    s_txt = encoder.encode_file(txt_file)

    assert s_py.shape == (256,)
    assert s_pdf.shape == (256,)
    assert s_txt.shape == (256,)

    # Verify that .py, .pdf, and .txt produce distinct sensory shapes
    sim_py_pdf = torch.cosine_similarity(s_py, s_pdf, dim=0).item()
    sim_py_txt = torch.cosine_similarity(s_py, s_txt, dim=0).item()

    assert sim_py_pdf < 0.9, f"expected distinct signatures, got sim={sim_py_pdf}"
    assert sim_py_txt < 0.9, f"expected distinct signatures, got sim={sim_py_txt}"


def test_motor_decoder_action_matching() -> None:
    decoder = MotorDecoder(vector_dim=256)

    # Test RUN intent
    run_intent = decoder.affordances[ActionPrimitive.RUN] + 0.05 * torch.randn(256)
    action, score = decoder.decode_intent(run_intent)
    assert action == ActionPrimitive.RUN
    assert score > 0.70

    # Test READ intent
    read_intent = decoder.affordances[ActionPrimitive.READ] + 0.05 * torch.randn(256)
    action, score = decoder.decode_intent(read_intent)
    assert action == ActionPrimitive.READ
    assert score > 0.70


import sys


def test_sensory_motor_turn_loop_execution(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    runtime = BoundedFileRuntime(root=tmp_path, loom=loom, allowed_programs={"python3": sys.executable})

    # Create a test script inside workspace
    script_path = tmp_path / "test_app.py"
    script_path.write_text("print('sensory motor success')\n")

    loop = SensoryMotorTurnLoop(runtime=runtime, vector_dim=256)

    # Simulate Turn 1: Target file is test_app.py, intent is RUN
    turn_result = loop.run_turn(
        user_input="Run the app script",
        target_path=script_path,
        simulated_motor_intent=decoder_run_intent(loop.decoder),
    )

    assert turn_result["action"] == ActionPrimitive.RUN
    assert turn_result["executed"] is True
    assert "sensory motor success" in turn_result["stdout"]
    assert turn_result["return_sensory_vector"].shape == (256,)


def test_sensory_motor_model_training(tmp_path: Path) -> None:
    from habitus_ai.sensory_motor_loop import train_sensory_motor_model

    # Create dummy files for training dataset
    (tmp_path / "app.py").write_text("import sys\n")
    (tmp_path / "data.pdf").write_bytes(b"%PDF-1.5 header")
    (tmp_path / "notes.txt").write_text("some notes")

    ckpt_path = tmp_path / "model_ckpt.pt"
    history = train_sensory_motor_model(
        workspace_dir=tmp_path,
        checkpoint_path=ckpt_path,
        epochs=10,
        vector_dim=256,
    )

    assert ckpt_path.exists()
    assert len(history["loss"]) == 10
    assert history["loss"][-1] < history["loss"][0]  # Loss decreased


def decoder_run_intent(decoder: MotorDecoder) -> torch.Tensor:
    return decoder.affordances[ActionPrimitive.RUN].clone()
