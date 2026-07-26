from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.frozen_rollout import (
    ADAPTER_RELATIVE_PATH,
    ARMS,
    BATCH_RELATIVE_PATH,
    evaluate,
    prepare,
)


def _control(output: str, algorithm: str) -> str:
    return (
        f'output_dir = "{output}"\n'
        f'[algo]\ntype = "{algorithm}"\n'
        f'[train.env.algo]\ntype = "{algorithm}"\n'
    )


def test_prepare_copies_one_batch_and_limits_control_difference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    batch = source / BATCH_RELATIVE_PATH
    batch.parent.mkdir(parents=True)
    batch.write_bytes(b"frozen")
    configs = source / "configs"
    configs.mkdir()
    (configs / "trainer.toml").write_text(
        'output_dir = "old"\nmax_steps = 1\n',
        encoding="utf-8",
    )
    stock_control = source / "run_default" / "control" / "orch.toml"
    stock_control.parent.mkdir(parents=True)
    stock_control.write_text(_control("old/run_default", "grpo"), encoding="utf-8")
    redco_control = tmp_path / "redco.toml"
    redco_control.write_text(
        _control("other/run_default", "redco_noop"),
        encoding="utf-8",
    )

    root = tmp_path / "replay"
    manifest = prepare(source, redco_control, root)

    hashes = {
        (root / arm / BATCH_RELATIVE_PATH).read_bytes() for arm in ARMS
    }
    assert hashes == {b"frozen"}
    assert len(
        {manifest["arms"][arm]["batch_sha256"] for arm in ARMS}
    ) == 1
    assert 'type = "redco_noop"' in (
        root / "redco" / "run_default" / "control" / "orch.toml"
    ).read_text(encoding="utf-8")


def test_evaluate_requires_exact_metrics_and_adapter_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    source_hash = "7" * 64
    (root / "source-manifest.json").parent.mkdir(parents=True)
    for arm in ARMS:
        batch = root / arm / BATCH_RELATIVE_PATH
        batch.parent.mkdir(parents=True)
        batch.write_bytes(b"batch")
        adapter = root / arm / ADAPTER_RELATIVE_PATH
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"adapter")
        (root / arm / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "time": 123.0,
                    "perf/throughput": 10.0,
                    "optim/grad_norm": 0.25,
                    "loss/mean": 0.125,
                    "entropy/all/mean": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    from hashlib import sha256

    source_hash = sha256(b"batch").hexdigest()
    (root / "source-manifest.json").write_text(
        json.dumps({"source_batch_sha256": source_hash}),
        encoding="utf-8",
    )

    passing = evaluate(root, root / "result.json")
    assert passing["passed_frozen_rollout_gate"]

    (root / "redco" / "metrics.jsonl").write_text(
        json.dumps(
            {
                "optim/grad_norm": 0.25,
                "loss/mean": 0.126,
                "entropy/all/mean": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failing = evaluate(root, root / "result.json")
    assert not failing["passed_frozen_rollout_gate"]
