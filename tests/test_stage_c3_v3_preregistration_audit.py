import hashlib
import json
from pathlib import Path

from redco.analysis.stage_c3_v3_preregistration import (
    DECISION_SHA256,
    V2_BUNDLE_SHA256,
    audit,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v3_audit_accepts_fresh_powered_protocol(tmp_path: Path) -> None:
    production_v1 = json.loads(
        Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v1.json"
        ).read_text(encoding="utf-8")
    )
    rules = production_v1["frozen_metrics_and_decision"]
    decision_sha = hashlib.sha256(
        json.dumps(
            rules,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    source = tmp_path / "source.py"
    rendered = tmp_path / "run.toml"
    source.write_text("source\n", encoding="utf-8")
    rendered.write_text("config\n", encoding="utf-8")
    common = {"frozen_metrics_and_decision": rules}
    v1 = {
        **common,
        "design": {"runs": [{"seed": 9301}]},
    }
    v2 = {
        **common,
        "design": {"runs": [{"seed": 9401}]},
    }
    v3 = {
        **common,
        "design": {
            "runs": [
                {"seed": 9501},
                {"seed": 9502},
                {"seed": 9503},
                {"seed": 9504},
            ]
        },
        "v2_terminal_record": {
            "bundle_sha256": V2_BUNDLE_SHA256,
            "scientific_gate_evaluated": False,
            "scientific_arms_started": 0,
        },
        "execution": {
            "forced_integration_smoke": {"sampled_pass_conditions": 0},
            "exact_power_gate": {
                "expected_target_informative_groups_per_sliced_step_minimum": 5.0,
                "position": "before_every_scientific_arm",
            },
            "scientific_early_abort": {
                "sampling_dependent_rules": 0,
                "computed_sampling_false_abort_probability": 0.0,
            },
            "root_seed_contract": (
                "sha256(master_seed, task_name, stable_episode_address)"
            ),
            "within_episode_common_random_numbers": True,
        },
        "source": {
            "sha256": {
                source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        },
        "rendered_configs": {
            "sha256": {
                rendered.name: hashlib.sha256(rendered.read_bytes()).hexdigest(),
            }
        },
    }
    v1_path = tmp_path / "v1.json"
    v2_path = tmp_path / "v2.json"
    v3_path = tmp_path / "v3.json"
    _write(v1_path, v1)
    _write(v2_path, v2)
    _write(v3_path, v3)

    result = audit(v1_path, v2_path, v3_path, root=tmp_path)

    assert decision_sha == DECISION_SHA256
    assert result["passed"] is True
    assert result["checks"]["all_source_hashes_match"] is True
    assert result["checks"]["all_rendered_config_hashes_match"] is True


def test_v3_audit_rejects_sampled_scientific_abort(tmp_path: Path) -> None:
    v1_path = Path(
        "configs/stage-c3/credit-confusion-live-preregistration-v1.json"
    )
    v2_path = Path(
        "configs/stage-c3/credit-confusion-live-preregistration-v2.json"
    )
    v3 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3["design"]["runs"] = [
        {"seed": seed} for seed in (9501, 9502, 9503, 9504)
    ]
    v3["v2_terminal_record"] = {
        "bundle_sha256": V2_BUNDLE_SHA256,
        "scientific_gate_evaluated": False,
        "scientific_arms_started": 0,
    }
    v3["execution"] = {
        "forced_integration_smoke": {"sampled_pass_conditions": 0},
        "exact_power_gate": {
            "expected_target_informative_groups_per_sliced_step_minimum": 5.0,
            "position": "before_every_scientific_arm",
        },
        "scientific_early_abort": {
            "sampling_dependent_rules": 1,
            "computed_sampling_false_abort_probability": 0.1,
        },
        "root_seed_contract": (
            "sha256(master_seed, task_name, stable_episode_address)"
        ),
        "within_episode_common_random_numbers": True,
    }
    v3["source"] = {"sha256": {}}
    v3["rendered_configs"] = {"sha256": {}}
    v3_path = tmp_path / "v3.json"
    _write(v3_path, v3)

    result = audit(v1_path, v2_path, v3_path)

    assert result["passed"] is False
    assert (
        result["checks"]["scientific_abort_has_no_sampled_outcome_rules"]
        is False
    )
