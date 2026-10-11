#!/usr/bin/env python3
"""Training script for the 3-Stage Sensory-Motor model in Habitus AI.

Trains non-language SensoryEncoder shape projections using contrastive representation learning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitus_ai.sensory_motor_loop import train_sensory_motor_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Habitus AI 3-Stage Sensory-Motor Model")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/home/nemo/MightyMem"),
        help="Path to workspace directory containing target files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/nemo/MightyMem/scratch/sensory_motor_model.pt"),
        help="Path to output PyTorch model checkpoint (.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=256, help="Sensory vector dimension (default: 256)")

    args = parser.parse_args()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting Sensory-Motor Model Training...")
    print(f"   Workspace:  {args.workspace}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Epochs:     {args.epochs}")
    print(f"   Vector Dim: {args.dim}")

    history = train_sensory_motor_model(
        workspace_dir=args.workspace,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        vector_dim=args.dim,
    )

    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]

    print(f"\n✅ Training Complete!")
    print(f"   Initial Loss: {initial_loss:.6f}")
    print(f"   Final Loss:   {final_loss:.6f}")
    print(f"   Saved Checkpoint: {history['checkpoint']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
