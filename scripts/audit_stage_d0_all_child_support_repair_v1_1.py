"""Audit the bounded pre-address Stage D support repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, amendment_path: Path) -> dict[str, Any]:
    amendment = _load(root / amendment_path)
    base_path = root / amendment["base_protocol"]
    base = _load(base_path)
    evidence_path = root / amendment["failure_evidence"]["artifact"]
    generated_path = root / "reports/stage-d0-all-child-live-runner-generation-v1-1.json"
    generated = _load(generated_path)
    verify_signed_payload(generated)

    with tarfile.open(evidence_path, "r:gz") as archive:
        names = [member.name.replace("\\", "/") for member in archive.getmembers()]
        stderr_names = [name for name in names if name.endswith("fixture-dry-run.json.stderr")]
        if len(stderr_names) != 1:
            raise ValueError("evidence must contain exactly one fixture dry-run stderr")
        handle = archive.extractfile(stderr_names[0])
        if handle is None:
            raise ValueError("fixture dry-run stderr is unreadable")
        dry_run_stderr = handle.read().decode("utf-8")

    merged_sources = dict(base["source_sha256"])
    merged_sources.update(amendment["source_sha256_overrides"])
    merged_sources.update(amendment["repair_source_sha256"])
    source_results = {
        name: {
            "expected": expected,
            "actual": _sha256(root / name),
            "passes": _sha256(root / name) == expected,
        }
        for name, expected in merged_sources.items()
    }
    merged = dict(base)
    merged["source_sha256"] = merged_sources
    merged_bytes = (json.dumps(merged, sort_keys=True, separators=(",", ":")) + "\n").encode()
    wrapper = (root / amendment["runner"]).read_text(encoding="utf-8")
    generator = (root / "scripts/generate_stage_d_all_child_live_runner_v1_1.py").read_text(
        encoding="utf-8"
    )
    budget = amendment["budget"]
    checks = {
        "base_protocol_unchanged": _sha256(base_path) == amendment["base_protocol_sha256"],
        "evidence_exact": (
            _sha256(evidence_path) == amendment["failure_evidence"]["artifact_sha256"]
            and evidence_path.stat().st_size == amendment["failure_evidence"]["artifact_bytes"]
        ),
        "zero_address_evidence": (
            not any(name.endswith("ADDRESS_STARTED") for name in names)
            and not any(name.endswith("FIXTURE_REQUESTS_STARTED") for name in names)
            and amendment["failure_evidence"]["fixture_or_support_address_markers"] == 0
            and amendment["failure_evidence"]["fixture_or_support_run_eval_requests"] == 0
        ),
        "failure_is_argparse_only": (
            "invalid choice: 'successor_fixture'" in dry_run_stderr
            and "--split" in dry_run_stderr
            and amendment["repair"]["historical_run_feasibility_bytes_changed"] is False
            and amendment["repair"]["only_runtime_change"].startswith(
                "route the successor runner through a versioned CLI module"
            )
        ),
        "only_expected_override": set(amendment["source_sha256_overrides"])
        == {"tests/test_stage_d_all_child_live_protocol.py"},
        "all_sources_exact": all(result["passes"] for result in source_results.values()),
        "merged_protocol_exact": (
            hashlib.sha256(merged_bytes).hexdigest() == amendment["merged_protocol_sha256"]
        ),
        "generated_runner_exact": (
            generated["generated_sha256"] == amendment["generated_runner_sha256"]
            and generated["signed_payload_sha256"] == amendment["generated_runner_audit_signature"]
            and generated["passes"] is True
        ),
        "runtime_merge_is_explicit": (
            "source_sha256_overrides" in wrapper
            and "repair_source_sha256" in wrapper
            and "merged_protocol_sha256" in wrapper
            and "generate_stage_d_all_child_live_runner_v1_1.py" in wrapper
            and "OLD_PROTOCOL" in generator
            and "NEW_PROTOCOL" in generator
            and "OLD_MODULE" in generator
            and "NEW_MODULE" in generator
        ),
        "science_fields_unchanged": all(
            amendment["repair"][field] is False
            for field in (
                "task_loading_logic_changed",
                "prompt_or_renderer_changed",
                "seed_or_order_changed",
                "sampling_or_budget_changed",
                "threshold_or_metric_changed",
                "fixture_or_support_data_changed",
                "model_or_adapter_changed",
                "scientific_interpretation_changed",
            )
        ),
        "cumulative_budget_passes": (
            budget["first_pod_billed_lifetime_seconds_conservative"]
            + budget["remaining_total_lifetime_seconds"]
            == budget["original_total_lifetime_seconds"]
            and budget["first_pod_billed_usd"] + budget["maximum_time_limited_repair_cost_usd"]
            <= budget["original_total_cost_cap_usd"]
            and budget["wallet_before_repair_selection_usd"]
            - budget["maximum_time_limited_repair_cost_usd"]
            >= budget["science_plus_untouchable_reserve_usd"]
        ),
        "last_deployment_only": (
            amendment["deployment_rule"]["repair_deployments_remaining"] == 1
            and amendment["deployment_rule"]["no_retry_after_any_address_started"] is True
            and amendment["deployment_rule"]["no_further_repair_deployment"] is True
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-all-child-support-repair-audit-v1-1",
            "amendment": amendment_path.as_posix(),
            "amendment_sha256": _sha256(root / amendment_path),
            "source_results": source_results,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("configs/stage-d/stage-d0-all-child-support-repair-amendment-v1-1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.amendment)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
