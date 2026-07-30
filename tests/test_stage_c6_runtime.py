from pathlib import Path


def test_stage_c6_driver_freezes_symmetric_constraint_and_fresh_runs() -> None:
    driver = Path("scripts/run_stage_c6_campaign_v1.sh").read_text(
        encoding="utf-8"
    )
    configs = list(Path("configs/stage-c6/rendered-v1").glob("*.toml"))
    assert len(configs) == 9
    assert "confusion_irrelevant|broadcast|9901" in driver
    assert "confusion_lucky|broadcast|9904" in driver
    assert "python -m redco.analysis.stage_c6_live" in driver
    assert "support-verification.json" in driver
    assert "verify_stage_c5_constraint_smoke.py" in driver
    for config in configs:
        text = config.read_text(encoding="utf-8")
        assert text.count("env.constrained_root_routes = true") == 2
        assert (
            'name = "runs/stage-c6/selected-initialization-merged"' in text
        )
