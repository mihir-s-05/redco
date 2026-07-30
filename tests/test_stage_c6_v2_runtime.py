from pathlib import Path


def test_stage_c6_v2_driver_freezes_measurement_split_and_fresh_runs() -> None:
    driver = Path("scripts/run_stage_c6_campaign_v2.sh").read_text(
        encoding="utf-8"
    )
    configs = list(Path("configs/stage-c6/rendered-v2").glob("*.toml"))
    assert len(configs) == 9
    assert driver.index("verify-model-identity") < driver.index(
        "for replicate in 1 2 3"
    )
    assert driver.index("verify-replicates") < driver.index(
        "RUNTIME_POWER_PASS"
    )
    assert driver.index("CANONICAL_SUPPORT_PASS") < driver.index(
        "confusion_irrelevant|broadcast|9911"
    )
    assert "verify-runtime-support" in driver
    assert "score_stage_c6_canonical_transformers.py" in driver
    assert "python -m redco.analysis.stage_c6_v2_live" in driver
    assert "confusion_lucky|broadcast|9914" in driver
    assert "uv_binary=" in driver
    assert 'export PATH="$(dirname "$uv_binary"):$PATH"' in driver
    assert 'resume_evidence_root="${REDCO_RESUME_EVIDENCE_ROOT:-}"' in driver
    assert driver.index('cp -a "$resume_evidence_root/initialization') < (
        driver.index("STRUCTURAL_SMOKE_PASS")
    )
    assert "\npip " not in driver
    for config in configs:
        text = config.read_text(encoding="utf-8")
        assert text.count("env.constrained_root_routes = true") == 2
        assert (
            'name = "runs/stage-c6/selected-initialization-merged"' in text
        )
