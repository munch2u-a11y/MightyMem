"""A random-born non-language sensory trunk sharing native recurrent state.

There is no sensory text window or external encoder. One current typed byte
event updates the very same hidden tensor used by the language trunk.
"""

from __future__ import annotations

from typing import Any

from .developmental_cortex import CortexConfig, require_torch, torch


def sensory_transition(model: Any, previous: Any, field: Any, hidden: Any, *, sequence_lengths: Any = None) -> Any:
    if hasattr(model, "observe_numeric"):
        return model.observe_numeric(previous, field, hidden, sequence_lengths=sequence_lengths)
    return model.sensory_trunk(previous, field, hidden, sequence_lengths=sequence_lengths)


if torch is not None:
    class WorkspaceSensoryTrunk(torch.nn.Module):
        def __init__(self, config: CortexConfig):
            super().__init__()
            self.byte_embedding = torch.nn.Embedding(256, config.byte_embedding_width)
            self.boundary_embedding = torch.nn.Parameter(torch.empty(config.byte_embedding_width))
            self.byte_norm = torch.nn.LayerNorm(config.byte_embedding_width)
            self.input_norm = torch.nn.LayerNorm(config.byte_embedding_width + config.graph_width)
            self.recurrent = torch.nn.GRU(config.byte_embedding_width + config.graph_width, config.hidden_width,
                                          num_layers=config.recurrent_layers, batch_first=True)
            torch.nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.boundary_embedding, mean=0.0, std=0.02)
            for name, parameter in self.recurrent.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(parameter)
                else:
                    torch.nn.init.zeros_(parameter)

        def forward(self, previous_bytes: Any, field: Any, hidden: Any, *, sequence_lengths: Any = None) -> Any:
            if previous_bytes.ndim != 2 or torch.any(previous_bytes < -1) or torch.any(previous_bytes > 255):
                raise ValueError("sensory trunk accepts boundary plus raw byte IDs, not language subwords")
            batch, length = previous_bytes.shape
            embedded = self.byte_embedding(previous_bytes.clamp_min(0))
            embedded = torch.where((previous_bytes < 0).unsqueeze(-1), self.boundary_embedding, embedded)
            inputs = self.input_norm(torch.cat((self.byte_norm(embedded),
                                                field.unsqueeze(1).expand(batch, length, -1)), dim=-1))
            if sequence_lengths is not None:
                if (sequence_lengths.shape != (batch,) or torch.any(sequence_lengths < 1)
                        or torch.any(sequence_lengths > length)):
                    raise ValueError("sensory sequence lengths must fit the padded batch")
                inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, sequence_lengths.cpu(), batch_first=True,
                                                                enforce_sorted=False)
            _, following = self.recurrent(inputs, hidden)
            return following
else:
    class WorkspaceSensoryTrunk:
        def __init__(self, *_: Any, **__: Any):
            require_torch()
