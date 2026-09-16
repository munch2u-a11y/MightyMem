"""Fixed-size learned sensory and language states, with a matched coupled control.

Only numeric neural states persist. No observation bytes, field values, paths,
or language history are retained as a buffer or supplied again to inference.
"""

from __future__ import annotations

from typing import Any

from .developmental_cortex import CortexConfig, DIRECTION_IDS, require_torch, torch
from .language_vocabulary import LanguageVocabulary
from .lexical_cortex import LexicalRecurrentCortex


if torch is not None:
    class SensoryReservoirCortex(LexicalRecurrentCortex):
        def __init__(self, config: CortexConfig, vocabulary: LanguageVocabulary, *,
                     stream_normalization: bool, sensory_state_policy: str, sensory_readout_mode: str = "static",
                     language_input_skip: str = "none"):
            if sensory_state_policy not in {"protected", "coupled"}:
                raise ValueError("reservoir policy must be protected or coupled")
            if sensory_readout_mode not in {"static", "language_query", "sensory_query"}:
                raise ValueError("unknown sensory readout mode")
            if language_input_skip not in ("none", "mean_embedding") or (language_input_skip != "none" and
                    (sensory_state_policy != "protected" or sensory_readout_mode != "static")):
                raise ValueError("language input skip requires a protected static reservoir")
            super().__init__(config, vocabulary, state_readout=True,
                             stream_normalization=stream_normalization, sensory_trunk=True)
            self.sensory_state_policy = sensory_state_policy
            # Initialize after the entire existing body so all shared tensors
            # match its random initialization. Both policies have identical
            # trainable parameters, initialization, and fixed state capacity.
            self.sensory_readout = torch.nn.Linear(config.recurrent_layers * config.hidden_width,
                                                  config.graph_width, bias=False)
            self.sensory_fusion_gate = torch.nn.Linear(2 * config.hidden_width, config.hidden_width)
            torch.nn.init.zeros_(self.sensory_readout.weight)
            torch.nn.init.zeros_(self.sensory_fusion_gate.weight)
            torch.nn.init.zeros_(self.sensory_fusion_gate.bias)
            self.sensory_readout_mode = sensory_readout_mode
            if sensory_readout_mode != "static":
                width = config.recurrent_layers * config.hidden_width
                self.sensory_query_gate = torch.nn.Linear(width, width)
                # 2*sigmoid(0)=1: new readouts begin with exactly the old
                # static projection. Both query policies share all tensors.
                torch.nn.init.zeros_(self.sensory_query_gate.weight)
                torch.nn.init.zeros_(self.sensory_query_gate.bias)
            self.language_input_skip = language_input_skip
            if language_input_skip != "none":
                self.language_input_skip_projection = torch.nn.Linear(
                    config.byte_embedding_width, config.recurrent_layers * config.hidden_width)
                # Neutral native growth: all existing tensors and initial
                # event trajectories remain unchanged before training.
                torch.nn.init.zeros_(self.language_input_skip_projection.weight)
                torch.nn.init.zeros_(self.language_input_skip_projection.bias)

        def initial_hidden(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros(2 * self.config.recurrent_layers, batch_size, self.config.hidden_width,
                               device=device, dtype=dtype)

        def split_hidden(self, hidden: Any) -> tuple[Any, Any]:
            layers = self.config.recurrent_layers
            if hidden.ndim != 3 or hidden.shape[0] != 2 * layers or hidden.shape[2] != self.config.hidden_width:
                raise ValueError("reservoir hidden state has the wrong shape")
            return hidden[:layers], hidden[layers:]

        def retained_field(self, hidden: Any) -> Any:
            language, sensory = self.split_hidden(hidden)
            language = language.transpose(0, 1).flatten(1)
            sensory = sensory.transpose(0, 1).flatten(1)
            if self.sensory_readout_mode != "static":
                query = language if self.sensory_readout_mode == "language_query" else sensory
                sensory = sensory * (2 * torch.sigmoid(self.sensory_query_gate(query)))
            field = self.state_readout(language) + self.sensory_readout(sensory)
            if hasattr(self, "state_readout_norm"):
                field = self.state_readout_norm(field)
            return torch.tanh(field)

        def observe_numeric(self, previous: Any, field: Any, hidden: Any, *, sequence_lengths: Any = None) -> Any:
            language, sensory = self.split_hidden(hidden)
            following = self.sensory_trunk(previous, field, sensory, sequence_lengths=sequence_lengths)
            gate = torch.sigmoid(self.sensory_fusion_gate(torch.cat((language, following), dim=-1)))
            # New perception may influence language state without forcing it
            # to discard all previous language-derived information.
            updated_language = language + gate * (following - language)
            return torch.cat((updated_language, following), dim=0)

        def forward(self, previous_bytes: Any, graph_field_tensor: Any, direction_ids: Any, hidden: Any = None,
                    *, sequence_lengths: Any = None) -> tuple[dict[str, Any], Any]:
            if hidden is None:
                hidden = self.initial_hidden(previous_bytes.shape[0], device=self.byte_embedding.weight.device,
                                             dtype=self.byte_embedding.weight.dtype)
            language, sensory = self.split_hidden(hidden)
            outputs, following = super().forward(previous_bytes, graph_field_tensor, direction_ids, language,
                                                  sequence_lengths=sequence_lengths)
            if self.language_input_skip != "none":
                # This variant consumes whole language events, as the native
                # communication runner and event-conditioned trainer do. It
                # cannot silently reinterpret a flattened mixed-event stream.
                if direction_ids.ndim == 2:
                    if not torch.all(direction_ids == direction_ids[:, :1]):
                        raise ValueError("language input skip requires one direction per event")
                    event_directions = direction_ids[:, 0]
                else:
                    event_directions = direction_ids
                hear = event_directions == DIRECTION_IDS["hear"]
                if bool(hear.any()):
                    mask = previous_bytes >= 0  # exclude BOS and padding
                    if sequence_lengths is not None:
                        mask = mask & (torch.arange(previous_bytes.shape[1], device=previous_bytes.device).unsqueeze(0)
                                       < sequence_lengths.unsqueeze(1))
                    embedded = self.byte_embedding(previous_bytes.clamp(min=0))
                    if self.stream_normalization:
                        embedded = self.word_input_norm(embedded)
                    summary = (embedded * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
                    residual = torch.tanh(self.language_input_skip_projection(summary)).to(following.dtype)
                    active = hear & mask.any(1)
                    residual = residual * active.unsqueeze(-1)
                    following = following + residual.reshape(previous_bytes.shape[0], self.config.recurrent_layers,
                                                              self.config.hidden_width).transpose(0, 1)
            retained = sensory if self.sensory_state_policy == "protected" else following
            return outputs, torch.cat((following, retained), dim=0)
else:
    class SensoryReservoirCortex:
        def __init__(self, *_: Any, **__: Any):
            require_torch()
