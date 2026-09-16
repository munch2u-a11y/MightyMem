"""Admit one actual workspace observation into the native recurrent state.

Incoming observations use a typed binary alphabet, not language-tokenized
prose. An optional learned sensory trunk updates either a shared hidden state
or a distinct numeric reservoir; existing lineages retain their original HEAR
input path. None is an action planner, retrieval interface, or pretrained encoder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .developmental_cortex import CortexLineageError, DIRECTION_IDS, _serialize_hidden, torch
from .stateful_language import StatefulConversation, language_field, language_model_sha256, load_language_checkpoint
from .workspace_observations import SENSOR_SCHEMA, WorkspaceObservation, WorkspaceSensor
from .workspace_cortex import sensory_transition


@dataclass(frozen=True)
class ObservationTurn:
    turn: int
    observation_sha256: str
    input_byte_count: int
    state_before_sha256: str
    state_after_sha256: str
    state_bytes: int
    model_sha256: str
    observation_encoding: str = SENSOR_SCHEMA
    current_input_passes: int = 1
    native_direction: str = "hear"
    language_generated: bool = False
    weight_updates: int = 0
    transcript_records_used: int = 0
    system_prompt_used: bool = False
    retrieval_used: bool = False
    pretrained_model_used: bool = False


class GroundedConversation(StatefulConversation):
    """The same numeric conversation state, accepting passive sensory events."""

    def observe(self, observation: WorkspaceObservation) -> ObservationTurn:
        if self._closed:
            raise RuntimeError("conversation is closed")
        if not isinstance(observation, WorkspaceObservation):
            raise TypeError("observe requires a typed workspace observation")
        if language_model_sha256(self.model) != self.model_sha256:
            raise CortexLineageError("weights changed during this conversation")
        payload = observation.encode()
        vocabulary = getattr(self.model, "vocabulary", None)
        if vocabulary is not None and vocabulary.pieces[:256] != tuple(bytes((n,)) for n in range(256)):
            raise CortexLineageError("workspace observations require the native raw-byte alphabet")
        before, _ = _serialize_hidden(self.hidden)
        with torch.no_grad():
            hidden = self.hidden
            for start in range(0, len(payload), self.config.maximum_event_bytes):
                chunk = payload[start : start + self.config.maximum_event_bytes]
                if hasattr(self.model, "sensory_trunk"):
                    hidden = sensory_transition(
                        self.model,
                        torch.tensor([[-1, *chunk]]),
                        language_field(self.model, hidden),
                        hidden,
                    )
                else:
                    _, hidden = self.model(
                        torch.tensor([[-1, *chunk]]),
                        language_field(self.model, hidden),
                        torch.tensor([DIRECTION_IDS["hear"]]),
                        hidden,
                    )
        after = self._commit(hidden)
        return ObservationTurn(
            turn=self.turn, observation_sha256=hashlib.sha256(payload).hexdigest(),
            input_byte_count=len(payload), state_before_sha256=hashlib.sha256(before).hexdigest(),
            state_after_sha256=hashlib.sha256(after).hexdigest(), state_bytes=len(after),
            model_sha256=self.model_sha256,
            native_direction="workspace_sensory_trunk" if hasattr(self.model, "sensory_trunk") else "hear",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    event = parser.add_mutually_exclusive_group(required=True)
    event.add_argument("--observe", metavar="RELATIVE_PATH")
    event.add_argument("--once", metavar="CURRENT_MESSAGE")
    parser.add_argument("--maximum-output-bytes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.maximum_output_bytes is not None and args.maximum_output_bytes < 1):
        parser.error("thread count and any explicit output limit must be positive")
    if (args.observe is not None) != (args.workspace is not None):
        parser.error("--workspace is required only with --observe; replies cannot reread a workspace")
    model = load_language_checkpoint(args.checkpoint)
    torch.set_num_threads(args.threads)
    observation = (WorkspaceSensor(args.workspace)
                   .observe(args.observe) if args.observe is not None else None)
    with GroundedConversation(model, state_path=args.state) as conversation:
        receipt = (conversation.observe(observation) if observation is not None else
                   conversation.reply(args.once, maximum_bytes=args.maximum_output_bytes))
    print(json.dumps({"event": "observation" if observation is not None else "language", "receipt": asdict(receipt)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
