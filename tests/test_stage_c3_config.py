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
