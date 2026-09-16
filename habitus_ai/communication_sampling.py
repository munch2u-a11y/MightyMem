"""Explicit, training-only dialogue sampling; never an inference input source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import random
from typing import Any, Sequence


SAMPLING_POLICIES = ("uniform", "balanced_families", "allocated_families")


class DialogueSampler:
    """Use a caller-owned RNG so optimizer snapshots can restore exact draws.

    Balanced sampling gives each named family the same number of dialogues,
    not the same number of symbols or a guaranteed share of gradient magnitude.
    Pair it with dialogue-normalized prediction loss when equal example weight
    is intended. Families come from explicit teaching provenance, not text cues.
    """

    def __init__(self, dialogues: Sequence[Any], *, batch_size: int, policy: str = "uniform",
                 family_batch_counts: Mapping[str, int] | None = None) -> None:
        self.dialogues = tuple(dialogues)
        if not self.dialogues or type(batch_size) is not int or batch_size < 1:
            raise ValueError("sampling requires training dialogues and a positive integer batch size")
        if policy not in SAMPLING_POLICIES:
            raise ValueError("unsupported training sampling policy")
        if policy != "allocated_families" and family_batch_counts is not None:
            raise ValueError("explicit family counts require allocated_families sampling")
        if any(item.split != "train" for item in self.dialogues):
            raise ValueError("the sampler accepts training-split dialogues only")
        if len({item.dialogue_id for item in self.dialogues}) != len(self.dialogues):
            raise ValueError("sampling requires unique dialogue IDs")
        self.batch_size, self.policy = batch_size, policy
        families: dict[str, list[Any]] = defaultdict(list)
        if policy != "uniform":
            for item in self.dialogues:
                family = item.provenance.get("training_family")
                if not isinstance(family, str) or not family or family != family.strip():
                    raise ValueError("family sampling requires explicit nonempty training_family provenance")
                families[family].append(item)
            if policy == "balanced_families" and batch_size % len(families):
                raise ValueError("batch size must be divisible by the number of training families")
        self.families = tuple((name, tuple(pool)) for name, pool in sorted(families.items()))
        if policy == "allocated_families":
            if (not isinstance(family_batch_counts, Mapping) or set(family_batch_counts) != set(families)
                    or any(type(value) is not int or value < 1 for value in family_batch_counts.values())
                    or sum(family_batch_counts.values()) != batch_size):
                raise ValueError("explicit family counts must cover every family with positive integers summing to batch size")
            self.family_batch_counts = {name: family_batch_counts[name] for name, _ in self.families}
        else:
            self.family_batch_counts = {name: batch_size // len(self.families) for name, _ in self.families}

    def sample(self, rng: random.Random) -> tuple[Any, ...]:
        if self.policy == "uniform":
            # Preserve the existing trainer's choices and RNG advancement.
            return tuple(rng.choices(self.dialogues, k=self.batch_size))
        selected = [item for name, pool in self.families
                    for item in rng.choices(pool, k=self.family_batch_counts[name])]
        rng.shuffle(selected)
        return tuple(selected)

    def identity(self) -> dict[str, Any]:
        return {
            "policy": self.policy, "batch_size": self.batch_size,
            "family_pool_sizes": {name: len(pool) for name, pool in self.families},
            "family_dialogues_per_batch": dict(self.family_batch_counts) if self.families else None,
            "within_family_sampling": "uniform_with_replacement" if self.families else None,
        }
