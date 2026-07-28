from __future__ import annotations

from redco.analysis.stage_c_warmstart import (
    build_warmstart_dataset,
    select_warmstart_checkpoint,
)


def _cases() -> dict:
    return {
        "signed_payload_sha256": "abc",
        "cases": [
            {
                "case_id": "planted_needle:gamma",
                "probe_name": "planted_needle",
                "context_route": "gamma",
                "actions": [str(index) for index in range(8)],
                "prompt": "choose",
            },
            {
                "case_id": "spurious_correlation:gamma",
                "probe_name": "spurious_correlation",
                "context_route": "gamma",
                "actions": ["0", "1"],
                "prompt": "choose",
            },
        ],
    }


def test_warmstart_dataset_uses_disjoint_seeds_and_skips_flat_states() -> None:
    examples, manifest = build_warmstart_dataset(
        _cases(),
        exogenous_seeds=range(20_000, 20_004),
    )

    assert len(examples) == 4
    assert {example["teacher_action"] for example in examples} == {"5"}
    assert manifest["heldout_seed_overlap"] is False
    assert manifest["skipped_action_independent_states"] == 0
    assert manifest["included_probes"] == ["planted_needle"]


def _model(name: str, mass: float, greedy: str | None) -> dict:
    return {
        "name": name,
        "cases": [
            {
                "probe_name": "planted_needle",
                "full_vocab_action_probabilities_t2": {"5": mass},
                "greedy_allowed_action": greedy,
            }
        ],
    }


def test_selection_uses_earliest_checkpoint_meeting_frozen_bounds() -> None:
    report = select_warmstart_checkpoint(
        {
            "models": [
                _model("base", 0.03, "3"),
                _model("sft_step_1", 0.10, "3"),
                _model("sft_step_2", 0.17, "3"),
                _model("sft_step_3", 0.22, "5"),
            ]
        },
        minimum_needle_mass_t2=0.15,
        maximum_needle_mass_t2=0.25,
        maximum_needle_greedy_rate=0.5,
        branch_count=6,
        groups_per_step=8,
        minimum_expected_informative_groups=4.75,
    )

    assert report["status"] == "pass"
    assert report["selected"]["step"] == 2
    assert report["selected"][
        "expected_informative_groups_per_step_at_minimum_mass"
    ] > 4.75


def test_selection_fails_closed_when_support_is_too_low() -> None:
    report = select_warmstart_checkpoint(
        {"models": [_model("sft_step_1", 0.05, "3")]},
        minimum_needle_mass_t2=0.15,
        maximum_needle_mass_t2=0.25,
        maximum_needle_greedy_rate=0.5,
        branch_count=6,
        groups_per_step=8,
        minimum_expected_informative_groups=4.75,
    )

    assert report["status"] == "fail"
    assert report["selected"] is None
