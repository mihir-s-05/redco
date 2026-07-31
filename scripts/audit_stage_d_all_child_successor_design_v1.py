"""Audit the CPU-only Stage D all-child successor design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.integrations.signed_subprocess import (  # noqa: E402
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = proportion + z * z / (2 * trials)
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return (center - radius) / denominator


def _binomial_tail(n: int, threshold: int, probability: float) -> float:
    return sum(
        math.comb(n, successes) * probability**successes * (1 - probability) ** (n - successes)
        for successes in range(threshold, n + 1)
    )


def audit(design_path: Path) -> dict[str, Any]:
    design = _read_json(design_path)
    hash_failures = []
    for relative, expected in design["source_sha256"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            hash_failures.append({"path": relative, "expected": expected, "actual": actual})
    trace_diagnostic = _read_json(ROOT / design["history"]["retrospective_trace_diagnostic"])
    context_audit = _read_json(ROOT / design["runtime_capacity"]["reference_independent_audit"])
    verify_signed_payload(trace_diagnostic)
    verify_signed_payload(context_audit)
    fixture_manifest = _read_json(ROOT / design["fresh_fixture_battery"]["manifest"])
    extension_manifest = _read_json(ROOT / design["fresh_data"]["extension_manifest"])
    inference = (ROOT / design["runtime_capacity"]["config"]).read_text(encoding="utf-8")
    scaffold = (ROOT / design["shared_midpoint_scaffold"]["prompt"]).read_text(encoding="utf-8")
    all_child_source = (ROOT / "src/redco/analysis/stage_d_all_child_support.py").read_text(
        encoding="utf-8"
    )
    branch_cli = (ROOT / "scripts/run_stage_d_all_child_branch_group.py").read_text(
        encoding="utf-8"
    )
    branch_source = (ROOT / "src/redco/analysis/stage_d_all_child_branch_group.py").read_text(
        encoding="utf-8"
    )
    scorer_source = (ROOT / "scripts/score_stage_d_all_child_replay.py").read_text(encoding="utf-8")
    regression_source = (ROOT / "tests/test_stage_d_all_child_support.py").read_text(
        encoding="utf-8"
    )
    lower = _wilson_lower(58, 64)
    downstream_tail = _binomial_tail(8, 5, 0.808)
    checks = {
        "status_is_cpu_only": design["status"] == "cpu_ready_not_live_authorized",
        "all_source_hashes_exact": not hash_failures,
        "v4_10_remains_terminal": ("failed and terminal" in design["history"]["v4_10_disposition"]),
        "retrospective_diagnostic_passed": all(trace_diagnostic["checks"].values()),
        "retrospective_diagnostic_not_confirmatory": (
            "not confirmatory" in trace_diagnostic["scope"]
        ),
        "literal_midpoint_rule_frozen": (
            "midpoint = len(paper_text) // 2" in scaffold
            and "paper_text[:midpoint]" in scaffold
            and "paper_text[midpoint:]" in scaffold
        ),
        "midpoint_context_audit_passed": all(context_audit["checks"].values()),
        "context_maximum_matches_design": (
            context_audit["maximum_conservative_tokens"]
            == design["runtime_capacity"]["maximum_conservative_prompt_plus_completion_tokens"]
            == 7898
        ),
        "returning_root_bound_matches_design": (
            context_audit["conservative_final_root_prompt_plus_completion_tokens"]
            == design["runtime_capacity"][
                "maximum_conservative_returning_root_plus_completion_tokens"
            ]
            == 6621
            and context_audit["checks"]["worst_case_returning_root_fits_8192"] is True
        ),
        "inference_is_eager_8192": (
            "max_model_len = 8192" in inference and "enforce_eager = true" in inference
        ),
        "fixture_manifest_passed": all(fixture_manifest["checks"].values()),
        "extension_manifest_passed": all(extension_manifest["checks"].values()),
        "extension_split_sizes_exact": (
            extension_manifest["partitions"]["successor_support"]["papers"] == 64
            and extension_manifest["partitions"]["successor_science_train"]["papers"] == 16
            and extension_manifest["partitions"]["successor_science_eval"]["papers"] == 32
        ),
        "fresh_holdout_coverage_not_inspected": (
            extension_manifest["selection"][
                "fresh_reference_positions_or_midpoint_coverage_inspected"
            ]
            is False
            and design["fresh_data"]["fresh_reference_location_or_midpoint_coverage_inspected"]
            is False
        ),
        "all_child_candidate_set_excludes_outcomes": all(
            forbidden in design["target_contract"]["selector_forbidden_inputs"]
            for forbidden in (
                "child action tokens or text",
                "reference evidence",
                "reward",
                "branch alternatives",
                "downstream scores",
            )
        ),
        "all_child_runtime_path_exists": (
            "maximum_targets=None" in branch_source
            and "stage_d_all_child_branch_group" in branch_cli
            and 'add_argument("--precommit"' in branch_cli
            and "verify_canonical_precommit(trace_path, precommit)" in branch_source
            and "all-depth-one-precommitted-v1" in all_child_source
            and "paper_has_any_joint_target" in all_child_source
        ),
        "trace_precommit_replay_scorer_chain_enforced": all(
            token in all_child_source
            for token in (
                "verify_canonical_precommit",
                "verify_replay_chain",
                "verify_scorer_chain",
                "source_trace_sha256",
                "precommit_signed_payload_sha256",
                "candidate_set_sha256",
                "replay_signed_payload_sha256",
                "derive_branch_group_seeds",
            )
        )
        and all(
            token in scorer_source
            for token in (
                "verify_canonical_precommit",
                "verify_replay_chain",
                "replay_signed_payload_sha256",
            )
        ),
        "decision_unit_weight_preserved": (
            "DecisionUnitWeight" in design["target_contract"]["outer_weighting"]
            and 'denominator": count' in all_child_source
            and "exact_decision_unit_weight_contract" in all_child_source
            and "outer_decision_unit_weight_sum" in all_child_source
            and "all_committed_targets_eligible" in all_child_source
            and "No post-outcome renormalization" in design["target_contract"]["outer_weighting"]
        ),
        "adversarial_regressions_present": all(
            name in regression_source
            for name in (
                "test_noncanonical_subset_precommit_is_rejected",
                "test_branch_rejects_bad_precommit_before_model_calls",
                "test_replay_chain_rejects_duplicate_alternative_index",
                "test_one_ineligible_committed_target_fails_paper",
            )
        ),
        "paper_is_independent_unit": design["target_contract"]["inferential_unit"]
        == "unique paper, never child target",
        "support_rule_is_58_of_64": (
            design["confirmatory_support_rule"]["papers"] == 64
            and design["confirmatory_support_rule"]["required_paper_successes"] == 58
        ),
        "wilson_lower_exceeds_0808": lower > 0.808,
        "eight_group_tail_at_0808_at_least_095": downstream_tail >= 0.95,
        "no_live_authorization": not any(
            design["authorization"][field]
            for field in (
                "live_fixture_requests",
                "live_support_requests",
                "scientific_training_requests",
            )
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-successor-design-audit",
            "design": design_path.as_posix(),
            "design_sha256": _sha256(design_path),
            "audit_generator_sha256": _sha256(Path(__file__)),
            "hash_failures": hash_failures,
            "power": {
                "wilson_lower_58_of_64": lower,
                "probability_at_least_5_of_8_when_p_0808": downstream_tail,
            },
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.design)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit("Stage D all-child successor design audit failed")


if __name__ == "__main__":
    main()
