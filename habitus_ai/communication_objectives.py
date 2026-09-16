"""Explicit training-only normalization for unequal-length teaching dialogues."""

from __future__ import annotations

from typing import Any


PREDICTION_NORMALIZATIONS = ("token", "dialogue")


def prediction_average(weighted_losses: Any, symbol_weights: Any, *, normalization: str = "token") -> Any:
    """Reduce one loss numerator and denominator per complete dialogue.

    Direction weights (SPEAK 1.0, HEAR 0.15) are applied before this function.
    Token normalization preserves the original aggregate objective. Dialogue
    normalization gives each nonempty example equal total prediction weight,
    without duplicating examples or changing the separate stopping objective.
    """
    if weighted_losses.ndim != 1 or weighted_losses.shape != symbol_weights.shape:
        raise ValueError("prediction reduction requires matching per-dialogue vectors")
    if normalization not in PREDICTION_NORMALIZATIONS:
        raise ValueError("unknown prediction normalization")
    if bool((symbol_weights < 0).any()) or not bool((symbol_weights > 0).any()):
        raise ValueError("prediction reduction requires nonnegative weights and a nonempty example")
    if normalization == "token":
        return weighted_losses.sum() / symbol_weights.sum().clamp(min=1.0)
    active = symbol_weights > 0
    return (weighted_losses[active] / symbol_weights[active]).mean()
