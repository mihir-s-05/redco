from __future__ import annotations

import itertools
import json
from hashlib import sha256
from pathlib import Path

from redco.analysis.deterministic_replay import evaluate_stock_stage
from redco.analysis.frozen_rollout import (
    ADAPTER_RELATIVE_PATH,
    ARMS,
    BATCH_RELATIVE_PATH,
    evaluate,
    prepare,
)
from redco.analysis.noop_confirmation import (
    CONFIRMATION_SEEDS,
)
from redco.analysis.noop_confirmation import (
    evaluate as evaluate_confirmation,
)
from redco.analysis.stock_noise import RUN_NAMES, calibrate


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
        'output_dir = "old"\nmatmul_precision = "high"\nmax_steps = 1\n',
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
    manifest = prepare(
        source,
        redco_control,
        root,
        matmul_precision="highest",
    )

    hashes = {
        (root / arm / BATCH_RELATIVE_PATH).read_bytes() for arm in ARMS
    }
    assert hashes == {b"frozen"}
    assert len(
        {manifest["arms"][arm]["batch_sha256"] for arm in ARMS}
    ) == 1
    assert manifest["matmul_precision"] == "highest"
    assert 'matmul_precision = "highest"' in (
        root / "stock-a" / "configs" / "trainer.toml"
    ).read_text(encoding="utf-8")
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


def test_deterministic_stage_stops_before_redco_when_stock_differs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deterministic"
    for arm, grad_norm in (("stock-a", 0.25), ("stock-b", 0.250001)):
        batch = root / arm / BATCH_RELATIVE_PATH
        batch.parent.mkdir(parents=True)
        batch.write_bytes(b"batch")
        adapter = root / arm / ADAPTER_RELATIVE_PATH
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(arm.encode())
        (root / arm / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "optim/grad_norm": grad_norm,
                    "loss/mean": 0.125,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    (root / "source-manifest.json").write_text(
        json.dumps({"source_batch_sha256": sha256(b"batch").hexdigest()}),
        encoding="utf-8",
    )
    result = evaluate_stock_stage(root, root / "result.json")
    assert not result["passed_stock_determinism_stage"]
    assert result["conditional_stop_honored"]
    assert not result["redco_executed"]


def test_stock_noise_calibration_freezes_double_observed_maximum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stock-noise"
    for index, name in enumerate(RUN_NAMES):
        batch = root / name / BATCH_RELATIVE_PATH
        batch.parent.mkdir(parents=True)
        batch.write_bytes(b"batch")
        adapter = root / name / ADAPTER_RELATIVE_PATH
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(name.encode())
        (root / name / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "optim/grad_norm": 0.25 + index * 1e-6,
                    "loss/mean": 0.125,
                    "entropy/all/mean": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    (root / "source-manifest.json").write_text(
        json.dumps({"source_batch_sha256": sha256(b"batch").hexdigest()}),
        encoding="utf-8",
    )
    comparisons = [
        {
            "first": first,
            "second": second,
            "l2": float(index + 1) * 1e-4,
            "max_abs": float(index + 1) * 1e-5,
        }
        for index, (first, second) in enumerate(
            itertools.combinations(RUN_NAMES, 2)
        )
    ]
    (root / "adapter-pairwise.json").write_text(
        json.dumps({"comparisons": comparisons}),
        encoding="utf-8",
    )

    result = calibrate(root, root / "bounds.json")
    assert result["calibration_passed"]
    assert result["status"] == "frozen_for_unseen_confirmation"
    assert result["pairwise_comparisons"] == 28
    assert result["equivalence_margins"]["adapter_l2"] == 2 * 28e-4


def test_frozen_trainer_confirmation_applies_pairwise_noise_bounds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "confirmation"
    bounds = {
        "exact_invariant_metrics": ["loss/mean", "entropy/all/mean"],
        "equivalence_margins": {
            "grad_norm_absolute_difference": 2e-5,
            "adapter_l2": 0.004,
            "adapter_max_abs": 4e-5,
        },
    }
    bounds_path = tmp_path / "bounds.json"
    bounds_path.write_text(json.dumps(bounds), encoding="utf-8")
    source_hash = sha256(b"batch").hexdigest()
    manifest = {
        "source_batch_sha256": source_hash,
        "bounds_sha256": sha256(bounds_path.read_bytes()).hexdigest(),
    }
    root.mkdir()
    (root / "source-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    comparisons = []
    for seed in CONFIRMATION_SEEDS:
        name = f"pair-s{seed}"
        for arm, grad_norm in (("stock", 0.25), ("redco", 0.25001)):
            arm_root = root / name / arm
            batch = arm_root / BATCH_RELATIVE_PATH
            batch.parent.mkdir(parents=True)
            batch.write_bytes(b"batch")
            adapter = arm_root / ADAPTER_RELATIVE_PATH
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(arm.encode())
            (arm_root / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "step": 1,
                        "optim/grad_norm": grad_norm,
                        "loss/mean": 0.125,
                        "entropy/all/mean": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        comparisons.append(
            {
                "pair": name,
                "seed": seed,
                "l2": 0.002,
                "max_abs": 2e-5,
            }
        )
    (root / "adapter-pairwise.json").write_text(
        json.dumps({"comparisons": comparisons}),
        encoding="utf-8",
    )

    passing = evaluate_confirmation(root, bounds_path, root / "result.json")
    assert passing["passed_frozen_trainer_noise_transfer_gate"]
    assert all(pair["passed"] for pair in passing["pairs"].values())

    failing_metrics = root / "pair-s5104" / "redco" / "metrics.jsonl"
    failing_metrics.write_text(
        json.dumps(
            {
                "step": 1,
                "optim/grad_norm": 0.25001,
                "loss/mean": 0.126,
                "entropy/all/mean": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failing = evaluate_confirmation(root, bounds_path, root / "result.json")
    assert not failing["passed_frozen_trainer_noise_transfer_gate"]
    assert not failing["pairs"]["pair-s5104"]["exact_metrics_passed"]
