"""Generate and audit the bounded Stage D v4.10 inner runner."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

OLD_MIGRATION = 'cmp -s "$migration_report" "$run_root/fixture-migration-recomputed.json"'
NEW_MIGRATION = '''"$uv_bin" run --frozen python \\
  scripts/audit_signed_json_equivalence_v4_9.py \\
  --expected "$migration_report" \\
  --actual "$run_root/fixture-migration-recomputed.json" \\
  --output "$run_root/fixture-migration-equivalence-v4-9.json"'''
OLD_CONFIG = 'selected_config="configs/stage-d/stage-d0-scaffold-inference-sft-v4.toml"'
NEW_CONFIG = (
    'selected_config="configs/stage-d/'
    'stage-d0-scaffold-inference-sft-v4-eager.toml"'
)
OLD_TAIL = 'inherited_tail="scripts/run_stage_d0_scaffold_support_v4_6.sh"'
NEW_TAIL = 'inherited_tail="$REDCO_V4_10_EAGER_TAIL"'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(parent: Path, output: Path) -> dict:
    parent_text = parent.read_text(encoding="utf-8")
    replacements = {
        OLD_MIGRATION: NEW_MIGRATION,
        OLD_CONFIG: NEW_CONFIG,
        OLD_TAIL: NEW_TAIL,
    }
    generated_text = parent_text
    for old, new in replacements.items():
        if generated_text.count(old) != 1:
            raise ValueError(f"parent contract missing or duplicated: {old}")
        generated_text = generated_text.replace(old, new)
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
    checks = {
        "three_named_replacements_exact": all(
            generated_text.count(new) == 1 and old not in generated_text
            for old, new in replacements.items()
        ),
        "exactly_three_diff_hunks": sum(line.startswith("@@") for line in diff) == 3,
        "request_sentinels_unchanged": (
            parent_text.count("REQUESTS_STARTED")
            == generated_text.count("REQUESTS_STARTED")
            == 6
        ),
        "scientific_tail_source_preserved": (
            parent_text.count('source "$instrumented_tail"')
            == generated_text.count('source "$instrumented_tail"')
            == 1
        ),
        "archive_comparison_preserved": (
            parent_text.count('cmp -s "$archive_manifest"')
            == generated_text.count('cmp -s "$archive_manifest"')
            == 1
        ),
        "live_eval_commands_preserved": (
            parent_text.count("run_eval \\")
            == generated_text.count("run_eval \\")
            == 2
        ),
        "instrumentation_awk_preserved": (
            parent_text.count('/^start_inference\\(\\)/ { emit = 1 }')
            == generated_text.count('/^start_inference\\(\\)/ { emit = 1 }')
            == 2
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-10-generated-inner-runner",
            "parent": parent.as_posix(),
            "parent_sha256": _sha256(parent),
            "generated": output.as_posix(),
            "generated_sha256": _sha256(output),
            "replacements": replacements,
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
