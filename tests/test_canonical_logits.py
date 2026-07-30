import math

import pytest

from redco.analysis.canonical_logits import selected_logprobs


def test_selected_logprobs_use_full_vocabulary_temperature() -> None:
    logits = [0.0, 1.0, 2.0, -1.0]
    observed = selected_logprobs(logits, [1, 3], temperature=2.0)
    weights = [math.exp(value / 2.0) for value in logits]
    normalizer = math.fsum(weights)
    assert math.exp(observed[1]) == pytest.approx(weights[1] / normalizer)
    assert math.exp(observed[3]) == pytest.approx(weights[3] / normalizer)


def test_selected_logprobs_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        selected_logprobs([], [0], temperature=1.0)
    with pytest.raises(ValueError):
        selected_logprobs([0.0], [2], temperature=1.0)
    with pytest.raises(ValueError):
        selected_logprobs([0.0], [0], temperature=0.0)
