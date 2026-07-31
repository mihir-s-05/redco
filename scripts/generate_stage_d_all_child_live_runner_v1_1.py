"""Generate the bounded-repair Stage D runner without changing science."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from generate_stage_d_all_child_live_runner_v1 import generate as generate_v1

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

OLD_PROTOCOL = 'protocol="configs/stage-d/stage-d0-all-child-support-preregistration-v1.json"'
NEW_PROTOCOL = (
    'protocol="/workspace/redco/.runtime/stage-d-v4-8/all-child-support-repair-protocol-v1-1.json"'
)
OLD_MODULE = "redco_evidence_selection_v2.run_feasibility"
NEW_MODULE = "redco_evidence_selection_v2.run_feasibility_successor_v1"


def generate(parent: str) -> str:
    output = generate_v1(parent)
    if output.count(OLD_PROTOCOL) != 1:
        raise ValueError("frozen v1 protocol reference is not unique")
    output = output.replace(OLD_PROTOCOL, NEW_PROTOCOL)
    if output.count(OLD_MODULE) != 2:
        raise ValueError("frozen feasibility module references are not exact")
    return output.replace(OLD_MODULE, NEW_MODULE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.read_text(encoding="utf-8")
    generated = generate(parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated.encode("utf-8"))
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-live-runner-generation-v1-1",
            "parent": args.parent.as_posix(),
            "parent_sha256": hashlib.sha256(parent.encode("utf-8")).hexdigest(),
            "generated": args.output.as_posix(),
            "generated_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
            "base_generator": "scripts/generate_stage_d_all_child_live_runner_v1.py",
            "only_change": "use runtime-merged repair protocol with two source overrides",
            "unified_diff": list(
                difflib.unified_diff(parent.splitlines(), generated.splitlines(), lineterm="")
            ),
            "passes": True,
        }
    )
    atomic_write_json(args.report, report)


if __name__ == "__main__":
    main()
