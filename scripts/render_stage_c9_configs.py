"""Render the frozen Stage-C9 paired efficiency configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (10031, 10032, 10033)
ARMS = ("local-e1", "local-e2", "branch-global-e2", "stock")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


def _sliced_config(source: str, *, arm: str, seed: int) -> str:
    output = f"runs/stage-c9/practical-efficiency/confusion_redundant/{arm}-s{seed}"
    text = source
    text = _replace_once(
        text,
        'output_dir = "runs/stage-c6/credit-confusion-live-v3/confusion_redundant/sliced-s9923"',
        f'output_dir = "{output}"',
    )
    text = _replace_once(
        text,
        'name = "runs/stage-c6/selected-initialization-merged"',
        'name = "runs/stage-c9/selected-initialization-merged"',
    )
    text = _replace_once(
        text,
        'import_path = "prime_rl.trainer.rl.redco_loss.clean_decision_loss"',
        'import_path = "prime_rl.trainer.rl.redco_loss.clipped_node_loss"',
    )
    text = _replace_once(
        text,
        "kwargs = { kl_tau = 0.001 }",
        'kwargs = { clip_epsilon = 0.2, kl_tau = 0.001, ratio_mode = "exact_sequence" }',
    )
    text = _replace_once(
        text,
        "[trainer.optim]\nlr = 1e-5",
        (
            '[trainer.optim]\n'
            'type = "adamw"\n'
            'lr = 1e-5\n'
            'weight_decay = 0.01\n'
            'max_norm = 1.0\n\n'
            '[trainer.scheduler]\n'
            'type = "constant"'
        ),
    )
    credit_scope = "global_loo" if arm == "branch-global-e2" else "local_loo"
    text = _replace_once(
        text,
        'cost_penalty = 0\nreplay_mode = "sliced"',
        (
            'cost_penalty = 0\n'
            'replay_mode = "sliced"\n'
            f'branch_credit_scope = "{credit_scope}"'
        ),
    )
    if arm.endswith("-e2"):
        text = _replace_once(text, "max_steps = 6", "max_steps = 12")
        text = _replace_once(
            text,
            "strict_snapshot_batches = true",
            "strict_snapshot_batches = true\ntrain_batch_reuse = 2",
        )
    text = text.replace("9923000", str(seed * 1000))
    text = text.replace("9923500", str(seed * 1000 + 500))
    text = _replace_once(text, "seed = 9923", f"seed = {seed}")
    return text


def _stock_config(source: str, *, seed: int) -> str:
    arm = "stock"
    output = f"runs/stage-c9/practical-efficiency/confusion_redundant/{arm}-s{seed}"
    text = source
    text = _replace_once(
        text,
        'output_dir = "runs/stage-c6/credit-confusion-live-v3/confusion_redundant/broadcast-s9923"',
        f'output_dir = "{output}"',
    )
    text = _replace_once(
        text,
        'name = "runs/stage-c6/selected-initialization-merged"',
        'name = "runs/stage-c9/selected-initialization-merged"',
    )
    text = _replace_once(
        text,
        "[trainer.optim]\nlr = 1e-5",
        (
            '[trainer.optim]\n'
            'type = "adamw"\n'
            'lr = 1e-5\n'
            'weight_decay = 0.01\n'
            'max_norm = 1.0\n\n'
            '[trainer.scheduler]\n'
            'type = "constant"'
        ),
    )
    text = text.replace("9923000", str(seed * 1000))
    text = text.replace("9923500", str(seed * 1000 + 500))
    text = _replace_once(text, "seed = 9923", f"seed = {seed}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sliced-source",
        type=Path,
        default=Path(
            "configs/stage-c6/rendered-v3/confusion_redundant-sliced-s9923.toml"
        ),
    )
    parser.add_argument(
        "--stock-source",
        type=Path,
        default=Path(
            "configs/stage-c6/rendered-v3/confusion_redundant-broadcast-s9923.toml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/stage-c9/rendered-v1"),
    )
    args = parser.parse_args()
    sliced_source = args.sliced_source.read_text(encoding="utf-8")
    stock_source = args.stock_source.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for seed in SEEDS:
        for arm in ARMS:
            text = (
                _stock_config(stock_source, seed=seed)
                if arm == "stock"
                else _sliced_config(sliced_source, arm=arm, seed=seed)
            )
            output = args.output_dir / f"{arm}-s{seed}.toml"
            output.write_bytes(text.encode("utf-8"))
            manifests.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": output.as_posix(),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    smoke_seed = 10030
    smoke = _sliced_config(
        sliced_source,
        arm="local-e2",
        seed=smoke_seed,
    )
    smoke = _replace_once(smoke, "max_steps = 12", "max_steps = 2")
    smoke = _replace_once(
        smoke,
        (
            'output_dir = "runs/stage-c9/practical-efficiency/'
            'confusion_redundant/local-e2-s10030"'
        ),
        'output_dir = "runs/stage-c9/practical-efficiency/smoke/local-e2-s10030"',
    )
    smoke = _replace_once(smoke, "num_examples = 64", "num_examples = 16")
    smoke_output = args.output_dir / "smoke-local-e2-s10030.toml"
    smoke_output.write_bytes(smoke.encode("utf-8"))
    manifests.append(
        {
            "arm": "integration-smoke-local-e2",
            "seed": smoke_seed,
            "path": smoke_output.as_posix(),
            "sha256": hashlib.sha256(smoke.encode()).hexdigest(),
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment": "stage-c9-practical-efficiency",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "configs": manifests,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
