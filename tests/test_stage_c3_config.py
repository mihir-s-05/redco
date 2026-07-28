from pathlib import Path

import pytest

from redco.analysis.stage_c3_config import render


@pytest.mark.parametrize("arm", ["broadcast", "sliced"])
def test_stage_c3_config_render_is_complete(tmp_path: Path, arm: str) -> None:
    template = Path(f"configs/stage-c3/credit-confusion-{arm}-template.toml")
    output = tmp_path / f"{arm}.toml"

    manifest = render(
        template,
        output,
        arm=arm,
        probe="confusion_irrelevant",
        seed=9301,
        run_root="runs/stage-c3/test",
    )
    text = output.read_text(encoding="utf-8")

    assert "__" not in text
    assert 'probe_names = ["confusion_irrelevant"]' in text
    assert f"seed = {9301}" in text
    assert manifest["arm"] == arm
    assert len(str(manifest["sha256"])) == 64


def test_stage_c3_config_render_rejects_unregistered_probe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported probe"):
        render(
            Path("configs/stage-c3/credit-confusion-broadcast-template.toml"),
            tmp_path / "bad.toml",
            arm="broadcast",
            probe="planted_needle",
            seed=9301,
            run_root="runs/stage-c3/test",
        )


def test_stage_c3_smoke_is_one_step_broadcast(tmp_path: Path) -> None:
    output = tmp_path / "smoke.toml"
    manifest = render(
        Path("configs/stage-c3/credit-confusion-broadcast-template.toml"),
        output,
        arm="broadcast",
        probe="confusion_redundant",
        seed=9400,
        run_root="runs/stage-c3/credit-confusion-live-v2/smoke",
        smoke=True,
    )
    text = output.read_text(encoding="utf-8")

    assert "max_steps = 1" in text
    assert "interval = 1" in text
    assert "num_examples = 16" in text
    assert "max_steps = 36" not in text
    assert manifest["smoke"] is True


def test_stage_c3_smoke_rejects_sliced_arm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="broadcast arm"):
        render(
            Path("configs/stage-c3/credit-confusion-sliced-template.toml"),
            tmp_path / "bad-smoke.toml",
            arm="sliced",
            probe="confusion_redundant",
            seed=9400,
            run_root="runs/stage-c3/credit-confusion-live-v2/smoke",
            smoke=True,
        )
