from __future__ import annotations

from redco.analysis.stage_c_warmstart_gate import _merge_equivalence


def _raw(name: str, probability: float, greedy: int = 20) -> dict:
    return {
        "models": [
            {
                "name": name,
                "cases": [
                    {
                        "case_id": "planted_needle:gamma",
                        "greedy_token_id": greedy,
                        "full_vocab_action_probabilities_t1": {"5": probability},
                        "full_vocab_action_probabilities_t2": {"5": probability},
                    }
                ],
            }
        ]
    }


def test_merge_equivalence_accepts_numerical_roundoff() -> None:
    result = _merge_equivalence(
        _raw("sft_step_2", 0.2),
        _raw("merged", 0.200001),
        selected_name="sft_step_2",
        tolerance=2e-5,
    )

    assert result["pass"]
    assert result["greedy_token_mismatches"] == 0


def test_merge_equivalence_rejects_greedy_mismatch() -> None:
    result = _merge_equivalence(
        _raw("sft_step_2", 0.2),
        _raw("merged", 0.2, greedy=21),
        selected_name="sft_step_2",
        tolerance=2e-5,
    )

    assert result["pass"] is False


def test_merge_equivalence_accepts_vllm_score_shape() -> None:
    def raw(name: str, probability: float) -> dict:
        row = {
            "case_id": "planted_needle:gamma",
            "greedy_token_id": 20,
            "action_probabilities": {"5": probability},
        }
        return {
            "models": [
                {
                    "name": name,
                    "temperatures": {
                        "1.0": [row],
                        "2.0": [row],
                    },
                }
            ]
        }

    result = _merge_equivalence(
        raw("sft_step_4", 0.09),
        raw("merged", 0.090001),
        selected_name="sft_step_4",
        tolerance=2e-5,
    )

    assert result["pass"]
