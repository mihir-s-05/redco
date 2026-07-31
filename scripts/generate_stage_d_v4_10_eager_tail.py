"""Generate and audit the Stage D v4.10 eager-runtime tail."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

OLD = '''      grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" \\
        "$run_root/inference-$label.log"
      return'''
NEW = '''      grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" \\
        "$run_root/inference-$label.log"
      grep -F "enforce_eager=True" \\
        "$run_root/inference-$label.log"
      if grep -Fq "Profiling CUDA graph memory" \\
        "$run_root/inference-$label.log"; then
        echo "eager runtime unexpectedly profiled CUDA graphs" >&2
        exit 1
      fi
      if grep -Fq "Capturing CUDA graphs" \\
        "$run_root/inference-$label.log"; then
        echo "eager runtime unexpectedly captured CUDA graphs" >&2
        exit 1
      fi
      touch "$run_root/EAGER_RUNTIME_PREFLIGHT_PASSED"
      return'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(parent: Path, output: Path) -> dict:
    parent_text = parent.read_text(encoding="utf-8")
    if parent_text.count(OLD) != 1:
        raise ValueError("parent must contain the frozen health-return block once")
    generated_text = parent_text.replace(OLD, NEW)
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
    parent_request_tail = parent_text[parent_text.index("run_eval() {") :]
    generated_request_tail = generated_text[generated_text.index("run_eval() {") :]
    checks = {
        "health_return_block_replaced_once": generated_text.count(NEW) == 1,
        "exactly_one_diff_hunk": sum(line.startswith("@@") for line in diff) == 1,
        "actual_eager_engine_arg_required": generated_text.count(
            'grep -F "enforce_eager=True"'
        )
        == 1,
        "cuda_graph_profile_and_capture_rejected": (
            generated_text.count('grep -Fq "Profiling CUDA graph memory"') == 1
            and generated_text.count('grep -Fq "Capturing CUDA graphs"') == 1
        ),
        "preflight_marker_written_once": generated_text.count(
            "EAGER_RUNTIME_PREFLIGHT_PASSED"
        )
        == 1,
        "preflight_marker_precedes_first_request_command": (
            generated_text.index("EAGER_RUNTIME_PREFLIGHT_PASSED")
            < generated_text.index("run_eval() {")
        ),
        "request_producing_tail_byte_identical": (
            parent_request_tail == generated_request_tail
        ),
        "inference_launch_byte_identical": (
            parent_text.count('inference @ "$config"')
            == generated_text.count('inference @ "$config"')
            == 1
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-10-eager-tail",
            "parent": parent.as_posix(),
            "parent_sha256": _sha256(parent),
            "generated": output.as_posix(),
            "generated_sha256": _sha256(output),
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
