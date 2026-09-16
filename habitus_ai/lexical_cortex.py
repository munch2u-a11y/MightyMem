"""The born-in recurrent body with a locally learned lexical alphabet.

This experimental shape does not change CortexConfig or the existing mind's
database/checkpoint format. Only new communication lineages may use it. The
vocabulary carries no neural weights, hidden history, or response selector.
"""

from __future__ import annotations

from typing import Any

from .developmental_cortex import BornInRecurrentCortex, CortexConfig, require_torch, torch
from .language_vocabulary import LanguageVocabulary


if torch is not None:
    class LexicalRecurrentCortex(BornInRecurrentCortex):
        def __init__(
            self, config: CortexConfig, vocabulary: LanguageVocabulary, *, state_readout: bool,
            stream_normalization: bool = False, sensory_trunk: bool = False,
        ):
            if type(stream_normalization) is not bool or (stream_normalization and not state_readout):
                raise ValueError("stream normalization requires a lexical state-readout lineage")
            if type(sensory_trunk) is not bool or (sensory_trunk and not state_readout):
                raise ValueError("a sensory trunk requires a lexical state-readout lineage")
            super().__init__(config)
            self.vocabulary = vocabulary
            self.stream_normalization = stream_normalization
            self.byte_embedding = torch.nn.Embedding(len(vocabulary), config.byte_embedding_width)
            self.byte_head = torch.nn.Linear(config.hidden_width, len(vocabulary))
            torch.nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)
            torch.nn.init.xavier_uniform_(self.byte_head.weight)
            torch.nn.init.zeros_(self.byte_head.bias)
            if state_readout:
                self.state_readout = torch.nn.Linear(config.recurrent_layers * config.hidden_width, config.graph_width)
            if stream_normalization:
                # Added after every random initializer so the shared initial
                # tensors match the unnormalized control at the same seed.
                self.word_input_norm = torch.nn.LayerNorm(config.byte_embedding_width)
                self.direction_input_norm = torch.nn.LayerNorm(config.direction_width)
                self.state_readout_norm = torch.nn.LayerNorm(config.graph_width)
            # Initialize last to preserve every shared random tensor of the
            # language-only control at the same initialization seed.
            if sensory_trunk:
                from .workspace_cortex import WorkspaceSensoryTrunk
                self.sensory_trunk = WorkspaceSensoryTrunk(config)

        def forward(
            self, previous_bytes: Any, graph_field_tensor: Any, direction_ids: Any, hidden: Any = None,
            *, sequence_lengths: Any = None,
        ) -> tuple[dict[str, Any], Any]:
            # Preserve the established recurrent body's shape and event
            # semantics, but never clamp a learned symbol to byte 255.
            if previous_bytes.ndim != 2:
                raise ValueError("previous symbols must have shape [batch, sequence]")
            if torch.any(previous_bytes < -1) or torch.any(previous_bytes >= len(self.vocabulary)):
                raise ValueError("symbol ID is outside the learned alphabet")
            batch, length = previous_bytes.shape
            lexical = self.byte_embedding(previous_bytes.clamp(min=0))
            boundary = self.boundary_embedding.view(1, 1, -1).expand(batch, length, -1)
            lexical = torch.where((previous_bytes < 0).unsqueeze(-1), boundary, lexical)
            graph_rows = graph_field_tensor.unsqueeze(1).expand(batch, length, -1)
            if direction_ids.shape == (batch,):
                directions = self.direction_embedding(direction_ids).unsqueeze(1).expand(batch, length, -1)
            elif direction_ids.shape == (batch, length):
                directions = self.direction_embedding(direction_ids)
            else:
                raise ValueError("direction_ids must have shape [batch] or [batch, sequence]")
            if self.stream_normalization:
                lexical = self.word_input_norm(lexical)
                directions = self.direction_input_norm(directions)
            inputs = self.input_norm(torch.cat((lexical, graph_rows, directions), dim=-1))
            if sequence_lengths is None:
                sequence, final_hidden = self.recurrent(inputs, hidden)
                final = sequence[:, -1, :]
            else:
                if sequence_lengths.shape != (batch,):
                    raise ValueError("sequence_lengths must have shape [batch]")
                if torch.any(sequence_lengths < 1) or torch.any(sequence_lengths > length):
                    raise ValueError("sequence lengths must be within the padded sequence")
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    inputs, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False,
                )
                packed_sequence, final_hidden = self.recurrent(packed, hidden)
                sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    packed_sequence, batch_first=True, total_length=length,
                )
                final = final_hidden[-1]
            return {
                # Historical head name; this shape predicts lexical symbols,
                # so reports must not label its loss as byte cross entropy.
                "byte_logits": self.byte_head(sequence),
                "stop_logits": self.stop_head(sequence).squeeze(-1),
                "route": torch.tanh(self.route_head(final)),
                "consequence": torch.tanh(self.consequence_head(final)).squeeze(-1),
                "valence": torch.tanh(self.valence_head(final)).squeeze(-1),
                "action_logits": self.action_head(final),
            }, final_hidden
else:
    class LexicalRecurrentCortex:
        def __init__(self, *_: Any, **__: Any):
            require_torch()
