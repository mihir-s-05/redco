from __future__ import annotations

from pathlib import Path

ROUTE_TOKEN_IDS = "[[7141,19127,32214,20255]]"


def test_stage_c6_v3_configs_enable_exact_root_categorical_everywhere() -> None:
    configs = sorted(Path("configs/stage-c6/rendered-v3").glob("*.toml"))
    manifests = sorted(
        Path("configs/stage-c6/rendered-v3").glob("*.manifest.json")
    )
    assert len(configs) == 9
    assert len(manifests) == 9

    for config in configs:
        text = config.read_text(encoding="utf-8")
        assert text.count('fused_lm_head_token_chunk_size = "disabled"') == 1
        assert text.count("[trainer.exact_categorical]") == 1
        assert text.count(f"token_groups = {ROUTE_TOKEN_IDS}") == 1
        assert text.count("env.constrained_root_routes = true") == 2
        assert (
            'name = "runs/stage-c6/selected-initialization-merged"' in text
        )
        assert "credit-confusion-live-v3" in text
        if config.name.startswith("structural-"):
            assert text.count("enable_token_export = true") == 1
        else:
            assert "enable_token_export" not in text


def test_prime_patch_contains_exact_normalizer_and_gradient_regression() -> None:
    patch = Path("patches/prime-rl-redco-stage-c6-v3.patch").read_text(
        encoding="utf-8"
    )
    assert "class ExactCategoricalConfig" in patch
    assert "def apply_exact_categorical_normalization" in patch
    assert "test_exact_categorical_cpu.py" in patch
    assert "group normalizer" in patch


def test_stage_c6_v3_driver_reads_multi_run_token_export_path() -> None:
    driver = Path("scripts/run_stage_c6_campaign_v3.sh").read_text(
        encoding="utf-8"
    )
    assert (
        '--token-exports "$structural_output/run_default/token_exports"'
        in driver
    )
