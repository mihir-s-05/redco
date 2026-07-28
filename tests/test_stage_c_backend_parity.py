from redco.analysis.stage_c_backend_parity import compare_backends


def _hf(probability: float) -> dict:
    return {
        "models": [
            {
                "name": name,
                "cases": [
                    {
                        "case_id": "needle:gamma",
                        "greedy_token_id": 20,
                        "full_vocab_action_probabilities_t1": {
                            "5": probability * multiplier
                        },
                        "full_vocab_action_probabilities_t2": {
                            "5": probability * multiplier
                        },
                    }
                ],
            }
            for name, multiplier in (("base", 1.0), ("adapter", 2.0))
        ]
    }


def _vllm(probability: float, *, adapter_multiplier: float = 2.0) -> dict:
    def rows(multiplier: float) -> list[dict]:
        return [
            {
                "case_id": "needle:gamma",
                "greedy_token_id": 20,
                "action_probabilities": {"5": probability * multiplier},
            }
        ]

    return {
        "models": [
            {
                "name": "base",
                "temperatures": {"1.0": rows(1.0), "2.0": rows(1.0)},
            },
            {
                "name": "adapter",
                "temperatures": {
                    "1.0": rows(adapter_multiplier),
                    "2.0": rows(adapter_multiplier),
                },
            },
        ]
    }


def test_exact_backend_parity_passes() -> None:
    report = compare_backends(
        _hf(0.1),
        _vllm(0.1),
        model_names=("base", "adapter"),
        maximum_logprob_difference=0.1,
    )
    assert report["status"] == "pass"
    assert report["observed"]["comparison_count"] == 4


def test_adapter_relative_backend_mismatch_fails() -> None:
    report = compare_backends(
        _hf(0.1),
        _vllm(0.1, adapter_multiplier=3.0),
        model_names=("base", "adapter"),
        maximum_logprob_difference=0.1,
    )
    assert report["status"] == "fail"
    assert report["observed"]["maximum_adapter_relative_logprob_difference"] > 0.1
