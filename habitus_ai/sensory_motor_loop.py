"""3-Stage Small -> Big -> Small Sensory-Motor Turn Loop module for Habitus AI.

Implements non-language continuous sensory vector shapes (file "feelings") and
motor intent vector decoding to achieve zero-schema, native action execution.
"""
from __future__ import annotations

import enum
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from context_loom.runtime import BoundedFileRuntime


class ActionPrimitive(str, enum.Enum):
    READ = "read"
    RUN = "run"
    WRITE = "write"
    PASS = "pass"


class SensoryEncoder(nn.Module):
    """Stage 1: Encodes raw file signatures into non-language 256-dim Sensory Shapes."""

    def __init__(self, vector_dim: int = 256) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        # Projection layer for byte n-gram histogram + structural metadata
        self.proj = nn.Linear(32, vector_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def encode_file(self, path: Path) -> torch.Tensor:
        """Encode file attributes and byte signature into a unit vector shape."""
        if not path.exists():
            # Default empty sensory vector
            return F.normalize(torch.zeros(self.vector_dim), dim=0)

        ext = path.suffix.lower()
        content_bytes = path.read_bytes()[:1024]  # Header slice

        # Feature vector (32-dim): extension hash + byte stats
        features = torch.zeros(32)
        ext_hash = int(hashlib.md5(ext.encode("utf-8")).hexdigest(), 16)
        features[0] = (ext_hash % 100) / 100.0

        # Byte histogram features (ASCII vs binary indicators)
        if content_bytes:
            features[1] = len(content_bytes) / 1024.0
            features[2] = sum(1 for b in content_bytes if 32 <= b <= 126) / max(1, len(content_bytes))  # ASCII ratio
            features[3] = sum(1 for b in content_bytes if b == 10 or b == 13) / max(1, len(content_bytes))  # Newline ratio
            features[4] = 1.0 if content_bytes.startswith(b"%PDF") else 0.0
            features[5] = 1.0 if b"import " in content_bytes or b"def " in content_bytes else 0.0

        sensory_shape = self.proj(features)
        sensory_shape = F.normalize(sensory_shape, dim=0)

        return sensory_shape


class MotorDecoder(nn.Module):
    """Stage 3: Decodes Motor Intent Vectors into Action Primitives via metric matching."""

    def __init__(self, vector_dim: int = 256) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.affordances: Dict[ActionPrimitive, torch.Tensor] = {}
        
        # Initialize distinct orthogonal vectors for each action primitive
        generator = torch.Generator().manual_seed(42)
        ortho = torch.randn(len(ActionPrimitive), vector_dim, generator=generator)
        ortho = F.normalize(ortho, dim=1)

        for idx, act in enumerate(ActionPrimitive):
            self.affordances[act] = ortho[idx]

    def decode_intent(self, intent_vector: torch.Tensor) -> Tuple[ActionPrimitive, float]:
        """Match intent vector against affordance vectors using cosine similarity."""
        intent_norm = F.normalize(intent_vector, dim=0)
        best_act = ActionPrimitive.PASS
        best_score = -1.0

        for act, affordance in self.affordances.items():
            sim = torch.cosine_similarity(intent_norm, affordance, dim=0).item()
            if sim > best_score:
                best_score = sim
                best_act = act

        return best_act, best_score


class SensoryMotorTurnLoop:
    """3-Stage Small -> Big -> Small Sensory-Motor Turn Loop Orchestrator."""

    def __init__(self, runtime: BoundedFileRuntime, vector_dim: int = 256) -> None:
        self.runtime = runtime
        self.vector_dim = vector_dim
        self.encoder = SensoryEncoder(vector_dim=vector_dim)
        self.decoder = MotorDecoder(vector_dim=vector_dim)

    def run_turn(
        self,
        user_input: str,
        target_path: Optional[Path] = None,
        simulated_motor_intent: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute one complete 3-stage turn cycle."""
        # Stage 1: Pre-Sensory Encoder (Small Model / Cortex)
        sensory_shape = (
            self.encoder.encode_file(target_path)
            if target_path
            else F.normalize(torch.zeros(self.vector_dim), dim=0)
        )

        # Stage 2: Conscious Mind (Big Model)
        # In a full deployment, this processes user_input + sensory_shape
        motor_intent = (
            simulated_motor_intent
            if simulated_motor_intent is not None
            else self.decoder.affordances[ActionPrimitive.PASS]
        )

        # Stage 3: Motor Decoder & Host Runtime Execution
        action, confidence = self.decoder.decode_intent(motor_intent)
        stdout = ""
        executed = False

        if action == ActionPrimitive.RUN and target_path:
            res = self.runtime.execute(
                "run-program",
                {"program": "python3", "arguments": [target_path.name]},
                scope="default",
            )
            stdout = res.receipt.result.get("stdout", res.memory_record.content)
            executed = res.receipt.success
        elif action == ActionPrimitive.READ and target_path:
            res = self.runtime.execute(
                "read-file",
                {"path": target_path.name},
                scope="default",
            )
            stdout = res.receipt.result.get("content", res.memory_record.content)
            executed = res.receipt.success

        # Compute return sensory shape for the next turn (reafference)
        return_sensory_vector = F.normalize(
            sensory_shape + 0.1 * torch.randn(self.vector_dim), dim=0
        )

        return {
            "user_input": user_input,
            "action": action,
            "confidence": confidence,
            "executed": executed,
            "stdout": stdout,
            "sensory_shape": sensory_shape,
            "motor_intent": motor_intent,
            "return_sensory_vector": return_sensory_vector,
        }


def train_sensory_motor_model(
    workspace_dir: Path,
    checkpoint_path: Path,
    epochs: int = 10,
    vector_dim: int = 256,
) -> Dict[str, Any]:
    """Train the SensoryEncoder projection weights using contrastive shape loss."""
    encoder = SensoryEncoder(vector_dim=vector_dim)
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.01)

    loss_history: list[float] = []

    files = list(workspace_dir.glob("*"))
    if not files:
        files = [workspace_dir / "sample.txt"]
        files[0].write_text("sample content")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)

        shapes = [encoder.encode_file(f) for f in files if f.is_file()]
        if len(shapes) >= 2:
            # Contrastive loss: minimize similarity between distinct file shape vectors
            sim = torch.cosine_similarity(shapes[0], shapes[1], dim=0)
            loss = sim ** 2 + 0.1 * (1.0 - torch.norm(shapes[0]))

        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "vector_dim": vector_dim,
            "epochs": epochs,
        },
        checkpoint_path,
    )

    return {
        "loss": loss_history,
        "checkpoint": str(checkpoint_path),
    }

