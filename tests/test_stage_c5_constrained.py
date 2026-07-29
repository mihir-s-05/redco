from __future__ import annotations

import math

import pytest

from redco.analysis.stage_c5_constrained import (
    constrained_root_scores,
    constrained_route_distribution,
)


def _root_scores(*, second_divergence: bool = False) -> dict:
    route_tokens = {
        "alpha": 101,
        "beta": 102,
        "gamma": 103,
        "delta": 104,
    }
    route_logprobs = {
        "alpha": math.log(0.1),
        "beta": math.log(0.2),
        "gamma": math.log(0.3),
        "delta": math.log(0.4),
    }
    details = {}
    for index, route in enumerate(route_tokens):
        details[route] = [
            {"token_id": 1, "temperature_2_logprob": -0.01},
            {
                "token_id": route_tokens[route],
                "temperature_2_logprob": route_logprobs[route],
            },
            {
                "token_id": 200 + index if second_divergence else 2,
                "temperature_2_logprob": -0.02,
            },
        ]
    return {
        "analysis": "fixture",
        "signed_payload_sha256": "fixture",
        "temperature_2": {
            "route_sequence_probabilities": {
                "alpha": 0.05,
                "beta": 0.10,
                "gamma": 0.15,
                "delta": 0.20,
            },
            "route_sequence_logprobabilities": {
                "alpha": math.log(0.05),
                "beta": math.log(0.10),
                "gamma": math.log(0.15),
                "delta": math.log(0.20),
            },
            "token_details": details,
        },
    }


def test_constrained_distribution_uses_divergent_choice_token() -> None:
    probabilities = constrained_route_distribution(_root_scores())

    assert probabilities == pytest.approx(
        {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4}
    )


def test_constrained_root_scores_are_unsigned_analysis_copy() -> None:
    original = _root_scores()
    constrained = constrained_root_scores(original)

    assert "signed_payload_sha256" not in constrained
    assert constrained["temperature_2"]["route_sequence_probabilities"] == pytest.approx(
        {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4}
    )
    assert original["temperature_2"]["route_sequence_probabilities"]["alpha"] == 0.05


def test_constrained_distribution_rejects_more_than_one_divergence() -> None:
    with pytest.raises(ValueError, match="exactly one divergent"):
        constrained_route_distribution(_root_scores(second_divergence=True))
