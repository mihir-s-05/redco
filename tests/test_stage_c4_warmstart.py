from __future__ import annotations

import copy

from redco.analysis.stage_c4_warmstart import (
    DIGITS,
    SELECTION_THRESHOLDS,
    audit_factorized_dataset,
    build_factorized_dataset,
    evaluate_candidate,
    select_earliest_candidate,
)
from redco.integrations.stage_c_prompts import (
    ROUTES,
    stage_c_branch_prompt,
    stage_c_root_prompt,
)


def _action_scores(probability: float = 0.12) -> dict:
    remainder = (0.96 - probability) / 7
    action_probabilities = {digit: probability if digit == "5" else remainder for digit in DIGITS}
    rows = [
        {
            "case_id": f"confusion_redundant:{route}",
            "probe_name": "confusion_redundant",
            "context_route": route,
            "action_probabilities": action_probabilities,
        }
        for route in ROUTES
    ]
    return {
        "models": [
            {
                "name": "candidate",
                "temperatures": {"2.0": rows},
            }
        ],
        "source": {"cases_sha256": "action-cases"},
        "signed_payload_sha256": "action-scores",
    }


def _root_scores(
    probabilities: dict[str, float] | None = None,
) -> dict:
    return {
        "temperature_2": {
            "route_sequence_probabilities": probabilities or {route: 0.24 for route in ROUTES}
        },
        "source": {"cases_sha256": "root-cases"},
        "signed_payload_sha256": "root-scores",
    }


def test_prompt_contracts_are_exact() -> None:
    assert stage_c_root_prompt().endswith("one of: alpha, beta, gamma, delta.")
    assert stage_c_branch_prompt("<route>delta</route>", DIGITS).endswith(
        "retained as an invalid action and receives the failure reward."
    )


def test_factorized_dataset_is_exact_product_without_joint_supervision() -> None:
    examples, manifest = build_factorized_dataset()
    assert len(examples) == 40
    assert manifest["status"] == "passed"
    assert all(manifest["checks"].values())
    assert manifest["factorization"] == {
        "empirical_total_variation": 0.0,
        "empirical_mutual_information_nats": 0.0,
    }
    assert manifest["supervision"]["root_and_target_labels_in_same_example"] == 0


def test_factorized_dataset_audit_rejects_joint_imbalance() -> None:
    examples, _ = build_factorized_dataset()
    mutated = copy.deepcopy(examples)
    target = next(row for row in mutated if row["example_kind"] == "target_format")
    target["digit_label"] = "7"
    target["messages"][-1]["content"] = "7"
    report = audit_factorized_dataset(mutated)
    assert report["status"] == "failed"
    assert not report["checks"]["every_route_digit_pair_occurs_once"]
    assert not report["checks"]["empirical_joint_equals_product_exactly"]


def test_candidate_passes_buffered_selection_and_unchanged_v3_power() -> None:
    _, manifest = build_factorized_dataset()
    report = evaluate_candidate(
        step=3,
        action_scores=_action_scores(),
        root_scores=_root_scores(),
        dataset_manifest=manifest,
    )
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["campaign_power"]["status"] == "passed"
    assert report["selection_thresholds"] == SELECTION_THRESHOLDS


def test_candidate_rejects_concentrated_root_even_with_digit_support() -> None:
    _, manifest = build_factorized_dataset()
    report = evaluate_candidate(
        step=1,
        action_scores=_action_scores(),
        root_scores=_root_scores(
            {
                "alpha": 0.02,
                "beta": 0.02,
                "gamma": 0.90,
                "delta": 0.02,
            }
        ),
        dataset_manifest=manifest,
    )
    assert report["status"] == "failed"
    assert not report["checks"]["every_route_mass_at_least_0_05"]
    assert not report["checks"]["delta_route_mass_in_0_10_to_0_35"]
    assert not report["checks"]["maximum_route_mass_at_most_0_55"]


def test_selection_chooses_earliest_passing_report() -> None:
    reports = [
        {"step": 4, "status": "passed", "signed_payload_sha256": "four"},
        {"step": 1, "status": "failed", "signed_payload_sha256": "one"},
        {"step": 3, "status": "passed", "signed_payload_sha256": "three"},
    ]
    selection = select_earliest_candidate(reports)
    assert selection["status"] == "passed"
    assert selection["selected_step"] == 3
    assert selection["evaluated_steps"] == [1, 3, 4]
