"""Validate the complete Stage D SFT patch from a clean pinned Prime worktree."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)

EXPECTED_COMMIT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
EXPECTED_SOURCES = {
    "packages/prime-rl-configs/src/prime_rl/configs/sft.py": (
        "a248f662e473a0d63f5e11da06792818b813ff8891accc54231ee964de63b8b9"
    ),
    "packages/prime-rl-configs/src/prime_rl/configs/trainer.py": (
        "8181f8c99b0c1be0382f457203de9e295c31a852eb0eb5a8ab88f999a5bdce4b"
    ),
    "src/prime_rl/trainer/ckpt.py": (
        "96e6c8d8b9d916a15205658a4040bfe85fc8a5cd76d49c36cffae6ddbc0b01dc"
    ),
    "src/prime_rl/trainer/models/layers/lora/multi_linear.py": (
        "f1fd0d52e68e1e835300af38b5186cde264dc81b14f31954203b99631e6f46c9"
    ),
    "src/prime_rl/trainer/sft/data.py": (
        "958c749090071643f2ae305cd0ca73230bb3237c17452a117c7f926485968cc1"
    ),
}
EXPECTED_PATCH_SHA256 = (
    "d8468a19de6f8319651765778a8abffb2999b1c84076fa9815c116608cdf8d73"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(clean: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(clean), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1:
        raise ValueError(f"expected a Windows drive path, got {resolved}")
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{relative}"


def _run_clean_stack(
    *,
    root: Path,
    clean: Path,
    config: Path,
) -> dict[str, Any]:
    root_wsl = _to_wsl(root)
    clean_wsl = _to_wsl(clean)
    config_wsl = _to_wsl(config)
    python_check = """
import inspect
import tomllib
from pathlib import Path

import torch
import torch.nn as nn

from prime_rl.configs.sft import SFTConfig, SFTDataConfig
from prime_rl.trainer.models.layers.lora.base import (
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.trainer.sft.data import load_sft_dataset
import prime_rl.configs.sft as sft_module
import prime_rl.trainer.ckpt as ckpt_module
import prime_rl.trainer.sft.data as data_module

config_path = Path(CONFIG_PATH)
config = SFTConfig.model_validate(
    tomllib.loads(config_path.read_text(encoding="utf-8"))
)
assert config.ckpt is not None
assert config.ckpt.weights is not None
assert config.ckpt.weights.adapter_only is True
assert config.data.data_files == "datasets/stage-d/scaffold-sft-v2.jsonl"
sources = [
    inspect.getfile(sft_module),
    inspect.getfile(ckpt_module),
    inspect.getfile(data_module),
]
assert all(CLEAN_ROOT in source for source in sources), sources
dataset = load_sft_dataset(
    SFTDataConfig(
        name="json",
        data_files="datasets/stage-d/scaffold-sft-v2.jsonl",
        shuffle=False,
    )
)
assert len(dataset) == 32
set_lora_num_tokens(torch.tensor([1]), reset_reference=True)
set_multilora_scaling(torch.tensor([1.0]), reset_reference=True)
layer = MultiLoRALinear(
    nn.Linear(8, 8),
    rank=8,
    n_adapters=1,
    use_grouped_mm=True,
)
assert layer.use_grouped_mm is False
print("CLEAN_RUNTIME_CONTRACTS_OK")
""".replace("CONFIG_PATH", repr(config_wsl)).replace(
        "CLEAN_ROOT", repr(clean_wsl)
    )
    prefix = [
        "export UV_PROJECT_ENVIRONMENT=/home/mihir/.venvs/redco-prime-cpu",
        "export UV_CACHE_DIR=/home/mihir/.cache/uv-redco",
        (
            "export PYTHONPATH="
            f"{shlex.quote(clean_wsl + '/src')}:"
            f"{shlex.quote(clean_wsl + '/packages/prime-rl-configs/src')}"
        ),
        f"cd {shlex.quote(root_wsl)}",
    ]
    uv = "/home/mihir/.local/uv-latest/uv"
    runtime_command = shlex.join(
        [
            uv,
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            clean_wsl,
            "python",
            "-c",
            python_check,
        ]
    )
    dry_run_command = shlex.join(
        [
            uv,
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            clean_wsl,
            "sft",
            "@",
            config_wsl,
            "--dry-run",
        ]
    )
    command = "; ".join([*prefix, runtime_command, dry_run_command])
    result = subprocess.run(
        ["wsl", "bash", "-lc", f"set -euo pipefail; {command}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "runtime_contract_passed": (
            "CLEAN_RUNTIME_CONTRACTS_OK" in result.stdout
        ),
        "dry_run_passed": "Dry run complete" in result.stdout,
    }


def audit(
    root: Path,
    clean: Path,
    patch: Path,
    config: Path,
) -> dict[str, Any]:
    head = _git(clean, "rev-parse", "HEAD")
    status = _git(clean, "status", "--porcelain")
    reverse = _git(clean, "apply", "--reverse", "--check", str(patch))
    diff_check = _git(clean, "diff", "--check")
    observed_paths = sorted(
        line[3:] for line in status.stdout.splitlines() if len(line) >= 4
    )
    source_results = {
        relative: {
            "expected": expected,
            "actual": _sha256(clean / relative),
        }
        for relative, expected in EXPECTED_SOURCES.items()
    }
    stack = _run_clean_stack(root=root, clean=clean, config=config)
    checks = {
        "clean_base_commit_exact": (
            head.returncode == 0 and head.stdout.strip() == EXPECTED_COMMIT
        ),
        "only_dependency_closure_modified": (
            observed_paths == sorted(EXPECTED_SOURCES)
        ),
        "patch_bytes_exact": _sha256(patch) == EXPECTED_PATCH_SHA256,
        "patch_reverse_check_passes": reverse.returncode == 0,
        "patched_diff_has_no_whitespace_errors": diff_check.returncode == 0,
        "all_patched_source_hashes_exact": all(
            row["actual"] == row["expected"]
            for row in source_results.values()
        ),
        "real_config_and_runtime_contracts_pass": (
            stack["returncode"] == 0 and stack["runtime_contract_passed"]
        ),
        "real_sft_launcher_dry_run_passes": (
            stack["returncode"] == 0 and stack["dry_run_passed"]
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-prime-sft-runtime-patch-v2-audit",
            "clean_worktree": str(clean),
            "base_commit": head.stdout.strip(),
            "patch": patch.relative_to(root).as_posix(),
            "patch_sha256": _sha256(patch),
            "config": config.relative_to(root).as_posix(),
            "observed_modified_paths": observed_paths,
            "source_results": source_results,
            "clean_stack": stack,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--clean-worktree", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path("patches/prime-rl-stage-d-sft-runtime-v2.patch"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage-d/stage-d0-scaffold-sft-v4.toml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.root.resolve(),
        args.clean_worktree.resolve(),
        args.patch.resolve(),
        args.config.resolve(),
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
