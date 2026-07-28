"""Audit the Stage-C3 v2 preregistration before any live model call."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(v1_path: Path, v2_path: Path) -> dict[str, Any]:
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v1_rules = _canonical(v1["frozen_metrics_and_decision"])
    v2_rules = _canonical(v2["frozen_metrics_and_decision"])
    v1_seeds = {int(run["seed"]) for run in v1["design"]["runs"]}
    v2_seeds = {int(run["seed"]) for run in v2["design"]["runs"]}
    source_checks = {
        name: _sha256(Path(name)) == expected
        for name, expected in v2["source"]["sha256"].items()
    }
    checks = {
        "decision_rules_byte_identical_to_v1": v1_rules == v2_rules,
        "decision_rule_sha_matches_v1": (
            hashlib.sha256(v1_rules).hexdigest()
            == v2["decision_rule_identity"]["v1_canonical_sha256"]
        ),
        "decision_rule_sha_matches_v2": (
            hashlib.sha256(v2_rules).hexdigest()
            == v2["decision_rule_identity"]["v2_canonical_sha256"]
        ),
        "fresh_seed_block_exact": v2_seeds == {9401, 9402, 9403, 9404},
        "fresh_seed_block_disjoint_from_v1": v1_seeds.isdisjoint(v2_seeds),
        "v1_terminal_bundle_frozen": (
            v2["v1_terminal_record"]["bundle_sha256"]
            == "1a4c239da6c878a1513964be2041dcd9e73066cc567456de8c896d5b8ddb13d9"
        ),
        "v1_scientific_gate_not_evaluated": (
            v2["v1_terminal_record"]["scientific_gate_evaluated"] is False
        ),
        "smoke_precedes_scientific_runs": (
            v2["execution"]["smoke"]["position"]
            == "before_every_scientific_arm"
        ),
        "every_arm_has_automatic_first_batch_abort": (
            v2["execution"]["early_abort"]["scope"] == "smoke_and_all_eight_arms"
        ),
        "all_source_hashes_match": all(source_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c3-v2-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1",
        type=Path,
        default=Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v1.json"
        ),
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v2.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.v1, args.v2)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
