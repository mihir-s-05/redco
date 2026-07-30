from pathlib import Path
from runpy import run_path


def test_future_stage_c9_render_retains_all_six_scoring_checkpoints() -> None:
    renderer = run_path("scripts/render_stage_c9_configs.py")
    sliced_config = renderer["_sliced_config"]
    stock_config = renderer["_stock_config"]
    sliced = Path(
        "configs/stage-c6/rendered-v3/"
        "confusion_redundant-sliced-s9923.toml"
    ).read_text(encoding="utf-8")
    stock = Path(
        "configs/stage-c6/rendered-v3/"
        "confusion_redundant-broadcast-s9923.toml"
    ).read_text(encoding="utf-8")

    assert "keep_last = 6" in sliced_config(
        sliced, arm="local-e1", seed=10031
    )
    assert "keep_last = 6" in sliced_config(
        sliced, arm="local-e2", seed=10031
    )
    assert "keep_last = 6" in stock_config(stock, seed=10031)
