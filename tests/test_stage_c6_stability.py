from redco.analysis.stage_c6_stability import _constant_reward_gradient


def test_constant_reward_target_gradient_is_zero_for_nonuniform_policy() -> None:
    probabilities = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    gradient = _constant_reward_gradient(probabilities, reward=1.0)
    assert max(abs(value) for value in gradient) < 1e-15
