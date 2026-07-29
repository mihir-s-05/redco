"""Power analysis for the Stage-C4 v4 marginal-shaping design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

VALID_ROUTE_MASS_FLOOR = 0.92
TARGET_GROUPS = 8
BRANCH_GROUP_SIZE = 11
TARGET_INFORMATIVE_GROUPS_FLOOR = 5.5
INHERITED_MEAN_DIGIT_5_MASS = 0.06281332379766019
INHERITED_EXACT_EXPECTED_GROUPS = 4.17036640685199
V3_FINAL_MEAN_DIGIT_5_MASS = 0.09591576364199421
V3_FINAL_EXACT_EXPECTED_GROUPS = 4.938244994712684
UNIFORM_DIGIT_MASS = 0.125


def informative_group_probability(success_mass: float, size: int) -> float:
    """Return the probability that a Bernoulli group contains both outcomes."""
    if not 0.0 <= success_mass <= 1.0:
        raise ValueError("success mass must lie in [0, 1]")
    if size < 2:
        raise ValueError("group size must be at least two")
    return 1.0 - success_mass**size - (1.0 - success_mass) ** size


def expected_groups_for_uniform_digit_mass(
    digit_mass: float,
    *,
    valid_route_mass: float = VALID_ROUTE_MASS_FLOOR,
) -> float:
    """Compute expected informative groups under route-invariant digit mass."""
    return (
        TARGET_GROUPS
        * valid_route_mass
        * informative_group_probability(digit_mass, BRANCH_GROUP_SIZE)
    )


def minimum_uniform_digit_mass(
    *,
    valid_route_mass: float = VALID_ROUTE_MASS_FLOOR,
    target_groups: float = TARGET_INFORMATIVE_GROUPS_FLOOR,
) -> float:
    """Solve the lower success-mass root needed to meet the power floor."""
    low = 0.0
    high = 0.5
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if expected_groups_for_uniform_digit_mass(
            midpoint,
            valid_route_mass=valid_route_mass,
        ) >= target_groups:
            high = midpoint
        else:
            low = midpoint
    return high


def build_report() -> dict[str, Any]:
    """Build the signed CPU-only design disposition."""
    required_mass = minimum_uniform_digit_mass()
    checks = {
        "inherited_policy_is_below_unchanged_power_floor": (
            INHERITED_EXACT_EXPECTED_GROUPS
            < TARGET_INFORMATIVE_GROUPS_FLOOR
        ),
        "v3_final_policy_is_below_unchanged_power_floor": (
            V3_FINAL_EXACT_EXPECTED_GROUPS
            < TARGET_INFORMATIVE_GROUPS_FLOOR
        ),
        "uniform_reward_blind_digit_target_has_analytic_headroom": (
            expected_groups_for_uniform_digit_mass(UNIFORM_DIGIT_MASS)
            >= TARGET_INFORMATIVE_GROUPS_FLOOR
        ),
        "required_uniform_digit_mass_exceeds_v3_final_mean": (
            required_mass > V3_FINAL_MEAN_DIGIT_5_MASS
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-v4-marginal-shaping-design",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "frozen_gate": {
            "valid_route_mass_floor": VALID_ROUTE_MASS_FLOOR,
            "target_groups": TARGET_GROUPS,
            "branch_group_size": BRANCH_GROUP_SIZE,
            "expected_informative_groups_floor": (
                TARGET_INFORMATIVE_GROUPS_FLOOR
            ),
        },
        "measurements": {
            "inherited_mean_digit_5_mass": INHERITED_MEAN_DIGIT_5_MASS,
            "inherited_exact_expected_groups": (
                INHERITED_EXACT_EXPECTED_GROUPS
            ),
            "v3_final_mean_digit_5_mass": V3_FINAL_MEAN_DIGIT_5_MASS,
            "v3_final_exact_expected_groups": (
                V3_FINAL_EXACT_EXPECTED_GROUPS
            ),
            "minimum_uniform_digit_5_mass_at_route_floor": required_mass,
            "uniform_reward_blind_digit_mass": UNIFORM_DIGIT_MASS,
            "uniform_reward_blind_expected_groups_at_route_floor": (
                expected_groups_for_uniform_digit_mass(UNIFORM_DIGIT_MASS)
            ),
        },
        "disposition": {
            "root_only": (
                "Rejected: preserving the inherited target marginal preserves "
                "an observed 4.170 informative groups, below the unchanged "
                "5.5 selection floor. Any pass would have to rely on "
                "uncontrolled collateral target drift."
            ),
            "selected_design": (
                "Continue the exact reward-blind route-by-digit product corpus "
                "to 32 optimizer steps, score only frozen even checkpoints, "
                "and retain every existing selection threshold."
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
