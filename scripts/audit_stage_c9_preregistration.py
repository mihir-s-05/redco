"""Audit the frozen Stage-C9 practical-efficiency protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-c9/practical-efficiency-preregistration-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/stage-c9-practical-efficiency-preregistration-audit-v1.json"
        ),
    )
    args = parser.parse_args()
    protocol: dict[str, Any] = json.loads(
        args.protocol.read_text(encoding="utf-8")
    )
    manifest_path = Path(protocol["source"]["configuration_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["frozen_status"] = (
        protocol["status"] == "frozen_before_gpu_observation"
    )
    checks["fresh_seeds"] = protocol["fixed_design"]["seeds"] == [
        10031,
        10032,
        10033,
    ]
    checks["matched_calls"] = (
        protocol["fixed_design"]["training_policy_calls_per_run"] == 576
    )
    checks["k_unchanged_at_11"] = (
        protocol["fixed_design"]["branch_group_size"] == 11
    )
    checks["bounded_budget"] = (
        protocol["hardware"]["maximum_total_compute_usd"] == 4.5
        and protocol["hardware"]["maximum_hourly_rate_usd"] == 2.0
    )
    checks["no_a100_h100_spot_or_disk"] = all(
        value in protocol["hardware"]["forbidden"]
        for value in (
            "A100",
            "H100",
            "spot or preemptible compute",
            "persistent storage",
        )
    )
    checks["explicit_stop_after_bridge"] = (
        "Stop synthetic efficiency tuning"
        in protocol["repair_and_stop_policy"]["after_stage_c9"]
    )
    checks["no_composite"] = (
        "No post-hoc composite score"
        in protocol["frozen_decision_rules"]["reporting"]
    )

    source_hashes = protocol["source_sha256"]
    checks["all_source_hashes_match"] = all(
        Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    )
    manifest_rows = manifest["configs"]
    checks["thirteen_configs_frozen"] = len(manifest_rows) == 13
    checks["manifest_config_hashes_match"] = all(
        _sha256(Path(row["path"])) == row["sha256"]
        for row in manifest_rows
    )

    semantics = []
    for row in manifest_rows:
        config = tomllib.loads(Path(row["path"]).read_text(encoding="utf-8"))
        arm = row["arm"]
        if arm == "integration-smoke-local-e2":
            semantics.append(
                config["max_steps"] == 2
                and config["orchestrator"]["train_batch_reuse"] == 2
            )
            continue
        common = (
            config["orchestrator"]["max_off_policy_steps"] == 0
            and config["orchestrator"]["strict_snapshot_batches"] is True
            and config["trainer"]["optim"]["type"] == "adamw"
            and config["trainer"]["scheduler"]["type"] == "constant"
            and config["trainer"]["model"]["lora"]["rank"] == 32
            and config["inference"]["seed"] == row["seed"]
        )
        if arm == "stock":
            specific = (
                config["max_steps"] == 36
                and "train_batch_reuse" not in config["orchestrator"]
                and config["orchestrator"]["algo"]["type"] == "grpo"
            )
        else:
            expected_steps = 6 if arm == "local-e1" else 12
            expected_scope = (
                "global_loo"
                if arm == "branch-global-e2"
                else "local_loo"
            )
            specific = (
                config["max_steps"] == expected_steps
                and config["orchestrator"].get("train_batch_reuse", 1)
                == (1 if arm == "local-e1" else 2)
                and config["orchestrator"]["algo"]["branch_group_size"] == 11
                and config["orchestrator"]["algo"]["branch_credit_scope"]
                == expected_scope
                and config["trainer"]["loss"]["import_path"]
                == "prime_rl.trainer.rl.redco_loss.clipped_node_loss"
            )
        semantics.append(common and specific)
    checks["all_config_semantics_match"] = all(semantics)

    patch = Path(
        "patches/prime-rl-redco-stage-c9-practical-efficiency.patch"
    )
    reverse = subprocess.run(
        [
            "git",
            "-C",
            "external/prime-rl",
            "apply",
            "--reverse",
            "--check",
            str(Path("../..") / patch),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    checks["prime_patch_matches_worktree"] = reverse.returncode == 0

    result = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c9-preregistration-audit",
            "protocol": args.protocol.as_posix(),
            "passed": all(checks.values()),
            "checks": checks,
            "config_count": len(manifest_rows),
            "source_hash_count": len(source_hashes),
            "patch_check_stderr": reverse.stderr,
        }
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
