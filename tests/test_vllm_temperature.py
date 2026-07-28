import math

import pytest

from redco.analysis.vllm_temperature import retemper_selected_logprobs


def test_retemperature_recovers_exact_softmax_distribution() -> None:
    raw = {10: math.log(0.8), 11: math.log(0.2)}

    observed = retemper_selected_logprobs(
        raw,
        [10, 11],
        temperature=2.0,
    )

    assert math.exp(observed[10]) == pytest.approx(2.0 / 3.0)
    assert math.exp(observed[11]) == pytest.approx(1.0 / 3.0)


def test_retemperature_uses_full_vocab_normalizer_for_selected_tokens() -> None:
    raw = {10: math.log(0.5), 11: math.log(0.3), 12: math.log(0.2)}

    observed = retemper_selected_logprobs(raw, [12], temperature=1.0)

    assert math.exp(observed[12]) == pytest.approx(0.2)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf])
def test_retemperature_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        retemper_selected_logprobs(
            {10: 0.0},
            [10],
            temperature=temperature,
        )
