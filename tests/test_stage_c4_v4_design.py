from __future__ import annotations

import math
import tomllib
from pathlib import Path

from redco.analysis.stage_c4_v4_design import (
    TARGET_INFORMATIVE_GROUPS_FLOOR,
    UNIFORM_DIGIT_MASS,
    build_report,
    expected_groups_for_uniform_digit_mass,
    minimum_uniform_digit_mass,
)


def test_root_only_disposition_follows_unchanged_power_floor() -> None:
    report = build_report()
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["measurements"]["inherited_exact_expected_groups"] < 5.5
    assert report["measurements"]["v3_final_exact_expected_groups"] < 5.5


def test_required_digit_mass_is_solved_against_power_floor() -> None:
    required = minimum_uniform_digit_mass()
    assert math.isclose(
        expected_groups_for_uniform_digit_mass(required),
        TARGET_INFORMATIVE_GROUPS_FLOOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert required < UNIFORM_DIGIT_MASS


def test_v4_config_extends_horizon_without_relaxing_training_recipe() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs/stage-c4/factorized-warmstart-sft-v3.toml").open(
        "rb"
    ) as handle:
        v3 = tomllib.load(handle)
    with (root / "configs/stage-c4/factorized-warmstart-sft-v4.toml").open(
        "rb"
    ) as handle:
        v4 = tomllib.load(handle)

    assert v4["max_steps"] == 32
    assert v4["ckpt"]["interval"] == 2
    assert v4["ckpt"]["keep_last"] == 16
    assert v4["data"]["seed"] == 7203004
    assert v4["renderer"] == v3["renderer"] == {"name": "prime-qwen3"}
    assert v4["data"]["data_files"] == v3["data"]["data_files"]
    assert v4["optim"] == v3["optim"]
    assert v4["model"]["lora"] == v3["model"]["lora"]
