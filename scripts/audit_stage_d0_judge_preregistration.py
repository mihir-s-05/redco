from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "configs/stage-d/stage-d0-judge-audit-preregistration-v1.json"
        ),
    )
    parser.add_argument(
        "--hardware-amendment",
        type=Path,
        default=Path(
            "configs/stage-d/stage-d0-judge-audit-hardware-amendment-v1-1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    amendment = json.loads(args.hardware_amendment.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}

    reference = prereg["reference_lock"]
    checks["reference_lock"] = {
        "expected": reference["sha256"],
        "observed": sha256(Path(reference["path"])),
    }
    checks["reference_lock"]["passes"] = (
        checks["reference_lock"]["expected"]
        == checks["reference_lock"]["observed"]
    )

    source_mismatches = {}
    for path, expected in prereg["frozen_source_hashes"].items():
        observed = sha256(Path(path))
        if observed != expected:
            source_mismatches[path] = {"expected": expected, "observed": observed}
    checks["frozen_sources"] = {
        "files": len(prereg["frozen_source_hashes"]),
        "mismatches": source_mismatches,
        "passes": not source_mismatches,
    }

    resource = amendment["resource"]
    checks["hardware"] = {
        "amends_correct_protocol": amendment["amends"]
        == args.preregistration.as_posix(),
        "scientific_fields_unchanged": not amendment["scientific_fields_changed"],
        "non_spot": resource["is_spot"] is False,
        "two_48gb_gpus": (
            resource["gpu_count"] == 2 and "48GB" in resource["gpu_type"]
        ),
        "within_hourly_cap": resource["price_per_hour_usd"] <= 2.0,
    }
    checks["hardware"]["passes"] = all(checks["hardware"].values())

    checks["budget"] = {
        "judge_gate_within_d0_cap": (
            prereg["resource_policy"]["maximum_judge_gate_cost_usd"] <= 1.25
            and prereg["resource_policy"]["maximum_live_minutes"] <= 50
        ),
        "passes": True,
    }
    checks["budget"]["passes"] = checks["budget"]["judge_gate_within_d0_cap"]

    passes = all(check["passes"] for check in checks.values())
    result = {
        "schema_version": 1,
        "preregistration": {
            "path": args.preregistration.as_posix(),
            "sha256": sha256(args.preregistration),
        },
        "hardware_amendment": {
            "path": args.hardware_amendment.as_posix(),
            "sha256": sha256(args.hardware_amendment),
        },
        "checks": checks,
        "decision": "pass" if passes else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passes:
        raise SystemExit("Stage D0 judge preregistration audit failed")


if __name__ == "__main__":
    main()
