"""Temperature transforms for vLLM next-token log-probability payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def retemper_selected_logprobs(
    raw_logprobs: Mapping[int, float],
    selected_token_ids: Sequence[int],
    *,
    temperature: float,
) -> dict[int, float]:
    """Return selected log-probabilities under ``softmax(logits / temperature)``.

    vLLM's completion log-probability payload contains normalized raw-model
    log-probabilities, independent of the sampling temperature. Because a
    constant shift in logits cancels under softmax, the exact temperature
    distribution can be recovered from the complete vocabulary payload.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not raw_logprobs:
        raise ValueError("raw_logprobs cannot be empty")
    if not selected_token_ids:
        raise ValueError("selected_token_ids cannot be empty")
    missing = [token_id for token_id in selected_token_ids if token_id not in raw_logprobs]
    if missing:
        raise ValueError(f"selected tokens missing from raw logprobs: {missing}")

    scaled = [float(value) / temperature for value in raw_logprobs.values()]
    maximum = max(scaled)
    log_normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in scaled)
    )
    return {
        token_id: float(raw_logprobs[token_id]) / temperature - log_normalizer
        for token_id in selected_token_ids
    }
