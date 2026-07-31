"""Generate and audit the one-line Stage D v4.9 inner-runner repair."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

OLD = 'cmp -s "$migration_report" "$run_root/fixture-migration-recomputed.json"'
NEW = '''"$uv_bin" run --frozen python \\
  scripts/audit_signed_json_equivalence_v4_9.py \\
  --expected "$migration_report" \\
  --actual "$run_root/fixture-migration-recomputed.json" \\
  --output "$run_root/fixture-migration-equivalence-v4-9.json"'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(parent: Path, output: Path) -> dict:
    parent_text = parent.read_text(encoding="utf-8")
    if parent_text.count(OLD) != 1:
        raise ValueError("parent must contain the frozen cmp line exactly once")
    generated_text = parent_text.replace(OLD, NEW)
    if OLD in generated_text or generated_text.count(
        "audit_signed_json_equivalence_v4_9.py"
    ) != 1:
        raise ValueError("replacement was not exact")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated_text, encoding="utf-8", newline="\n")

    diff = list(
        difflib.unified_diff(
            parent_text.splitlines(),
            generated_text.splitlines(),
            fromfile=parent.as_posix(),
            tofile=output.as_posix(),
            lineterm="",
        )
    )
    hunks = [line for line in diff if line.startswith("@@")]
    checks = {
        "old_line_exactly_once": parent_text.count(OLD) == 1,
        "new_verifier_exactly_once": generated_text.count(
            "audit_signed_json_equivalence_v4_9.py"
        )
        == 1,
        "exactly_one_diff_hunk": len(hunks) == 1,
        "request_sentinels_unchanged": (
            parent_text.count("REQUESTS_STARTED")
            == generated_text.count("REQUESTS_STARTED")
            == 6
        ),
        "scientific_tail_unchanged": (
            parent_text.count('source "$instrumented_tail"')
            == generated_text.count('source "$instrumented_tail"')
            == 1
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-9-generated-inner-runner",
            "parent": parent.as_posix(),
            "parent_sha256": _sha256(parent),
            "generated": output.as_posix(),
            "generated_sha256": _sha256(output),
            "replacement": {"old": OLD, "new": NEW},
            "unified_diff": diff,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.parent, args.output)
    atomic_write_json(args.report, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
