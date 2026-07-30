"""Deterministic full-vocabulary transforms for canonical policy scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence


def selected_logprobs(
    logits: Sequence[float],
    selected_token_ids: Sequence[int],
    *,
    temperature: float,
) -> dict[int, float]:
    """Compute selected full-vocabulary log-probabilities deterministically."""
    if not logits:
        raise ValueError("logits cannot be empty")
    if not selected_token_ids:
        raise ValueError("selected_token_ids cannot be empty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if any(token_id < 0 or token_id >= len(logits) for token_id in selected_token_ids):
        raise ValueError("selected token id is outside the vocabulary")
    scaled = [float(value) / temperature for value in logits]
    if any(not math.isfinite(value) for value in scaled):
        raise ValueError("logits must be finite")
    maximum = max(scaled)
    log_normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in scaled)
    )
    return {
        token_id: scaled[token_id] - log_normalizer
        for token_id in selected_token_ids
    }
