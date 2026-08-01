"""Audit the frozen Stage-D E2 4B bridge protocol without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_training_bridge import SealedTrainingBatch
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def audit(protocol_path: Path) -> dict[str, Any]:
    protocol = _load_object(protocol_path)
    failures: list[str] = []
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        failures.append("source hash map is missing")
    else:
        for raw_path, expected in source_hashes.items():
            path = Path(raw_path)
            if not path.is_file() or _sha256(path.read_bytes()) != expected:
                failures.append(f"source hash mismatch: {raw_path}")

    frozen = protocol.get("frozen_input", {})
    golden_path = Path(str(frozen.get("golden_manifest", "")))
    batch_path = Path(str(frozen.get("sealed_batch", "")))
    if not golden_path.is_file() or _sha256(golden_path.read_bytes()) != frozen.get(
        "golden_manifest_sha256"
    ):
        failures.append("golden manifest hash mismatch")
    if not batch_path.is_file() or _sha256(batch_path.read_bytes()) != frozen.get(
        "sealed_batch_sha256"
    ):
        failures.append("sealed batch hash mismatch")
    if batch_path.is_file():
        batch = SealedTrainingBatch.verify_bytes(batch_path.read_bytes())
        if batch.training_batch_identity != frozen.get("training_batch_identity"):
            failures.append("training batch identity mismatch")
        if len(batch.records) != frozen.get("record_count"):
            failures.append("sealed batch record count mismatch")

    trainer_path = Path("configs/stage-d/stage-d-e2-trainer-v1.toml")
    control_path = Path("configs/stage-d/stage-d-e2-control-v1.toml")
    trainer = tomllib.loads(trainer_path.read_text(encoding="utf-8"))
    control = tomllib.loads(control_path.read_text(encoding="utf-8"))
    if trainer.get("max_steps") != 1 or control.get("max_steps") != 1:
        failures.append("Prime TOMLs do not freeze one step")
    if trainer.get("max_concurrent_runs") != 1:
        failures.append("trainer does not freeze one run")
    if trainer.get("loss", {}).get("kwargs", {}).get("kl_tau") != 0.0:
        failures.append("synthetic smoke KL coefficient is not zero")
    if trainer.get("model", {}).get("optim_cpu_offload") is not False:
        failures.append("optimizer CPU offload is not disabled")
    if trainer.get("ckpt", {}).get("weights", {}).get("adapter_only") is not True:
        failures.append("checkpoint is not adapter-only")

    runner = Path("scripts/run_stage_d_e2_4b_live_v1.sh").read_text(encoding="utf-8")
    required_runner_fragments = (
        "tests/test_stage_d_live_update_torch.py",
        "stage_d_e2_control.py parse-configs",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "--nproc-per-node=1",
        "REDCO_LIVE_UPDATE_BINDING",
        "REDCO_LIVE_UPDATE_RECEIPTS",
    )
    for fragment in required_runner_fragments:
        if fragment not in runner:
            failures.append(f"runner contract missing: {fragment}")

    hardware = protocol.get("hardware_policy", {})
    if (
        hardware.get("gpu_count") != 1
        or hardware.get("spot") is not False
        or hardware.get("persistent_storage") != "forbidden"
        or float(hardware.get("maximum_total_cost_usd", 99)) > 2.0
    ):
        failures.append("hardware policy exceeds the bounded smoke contract")
    authorization = protocol.get("authorization", {})
    if authorization.get("post_authorization") != (
        "no retry or second optimizer attempt under any failure"
    ):
        failures.append("post-authorization no-retry rule differs")

    payload = {
        "schema_version": 1,
        "analysis": "stage-d-e2-4b-preregistration-audit-v1",
        "protocol": protocol_path.as_posix(),
        "protocol_sha256": _sha256(protocol_path.read_bytes()),
        "source_hash_count": len(source_hashes) if isinstance(source_hashes, dict) else 0,
        "failures": failures,
        "decision": "pass" if not failures else "fail",
    }
    payload["audit_signature"] = _sha256(canonical_json(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol)
    args.output.write_bytes(canonical_json(result))
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
