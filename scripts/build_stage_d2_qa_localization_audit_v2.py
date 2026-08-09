"""Build the non-authorizing Phase-2 disposable-overlay QA audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

from verify_stage_d2_qa_localization_v2 import (
    EXPECTED_MODULE_BINDINGS,
    EXPECTED_OBSERVER_METHODS,
    EXPECTED_RUNTIME,
    POSITIVE_EVIDENCE_FIELDS,
    SAMPLING_CONTRACT_SHA256,
    TEST_NODES,
)

from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.contracts import canonical_json

ROOT = Path(__file__).parents[1].resolve()
PARENT_COMMIT = "1ae6822b840858e2a27daeb44596342f2000979e"
PARENT_TREE = "38e206f4f88b8eed014a42532969e4a4835ff9a5"
AUDIT_RELATIVE = Path("reports/stage-d2-qa-localization-audit-v2.json")
V1_AUDIT_RELATIVE = Path("reports/stage-d2-qa-localization-audit-v1.json")
V1_AUDIT_SHA256 = "9b6d9b1db80da5ce8c8d344923c1a009e0f13e7fa53c5df150eb556f3b684baf"
REVIEWED_EVIDENCE_SHA256 = (
    "668fcdd56f49a283933b38016fec5a138b30b2f1c9c353f7be8d83ad4e4ab76f"
)
REVIEWED_RECEIPT_SHA256 = (
    "b386d4c0d5b7186bb0bffb73be4f04aaab367e5bc106eb1d98d4fcf822a2842f"
)
REVIEWED_REPORT_SHA256 = (
    "af3ead6ffebae0efbec434ee026fa8f2c60a352cf4c0ab4979541c8f876a89d0"
)
BOUND_PATHS = (
    "scripts/build_stage_d2_qa_localization_audit_v2.py",
    "scripts/verify_stage_d2_qa_localization_v2.py",
    "src/redco/analysis/stage_d_dependency_stack.py",
    "tests/test_stage_d_source_env_pinned.py",
)
EVIDENCE_FIELDS = {
    "schema_version",
    "domain",
    "parent_commit",
    "parent_tree",
    "dependency_authentication_runs",
    "dependency_stack",
    "abi_probe",
    "runtime",
    "tests",
    "positive",
    "external_activity",
}
ABI_FIELDS = {
    "schema_version",
    "domain",
    "python",
    "split_engine_sampling",
    "sampling_contract_sha256",
    "sampling_field_count",
    "episode_trace_model_call_v2_validated",
    "v2_provenance_record_count",
    "prepared_observer_methods",
    "controller_watchdog_interface",
    "module_bindings",
}
MODULE_BINDING_FIELDS = {"module", "relative_path", "sha256", "bytes"}
TEST_FIELDS = {"nodes", "collected", "passed", "failed", "skipped", "xfail"}
EXTERNAL_ACTIVITY_FIELDS = {
    "prime_calls",
    "provider_calls",
    "model_calls",
    "gpu_calls",
    "wallet_calls",
    "external_network_calls",
    "loopback_fixture_calls",
    "qasper_rows_read",
    "parquet_access",
    "ordinal_181_accessed",
    "launch_dataset_reads",
    "external_prime_rl_imported",
    "external_prime_rl_modified",
}
EXPECTED_POSITIVE_HASHES = {
    "source_sha256": "4371eaa711ac378fda81a8b059b8dd78cc5029bc25688ff5acb7c5cd1e7059ff",
    "trace_sha256": "254cfacf953dedd6db441404c8e4d6007c1c3b6783ed63f3c0ef2179bddadeb5",
    "recorded_action_digest": (
        "4faf683c2c2fedf3680b2ffec07c42435b122d2cdb529ba66506fd28c1d46f31"
    ),
    "commitment_receipt_sha256": (
        "7d306d0815fdd6f5dc52b2e8433a1106e9825b9a2397f68a9c17a49c9e7be74c"
    ),
    "correspondence_receipt_sha256": (
        "86f90882d4e915a7024dcffe62e413571fe9bc85ed6a980b92977da1040a39c9"
    ),
    "roster_sha256": "e4b97acc0be6a5bd4690898ae48270a082304d4131f2a143c26b465d917a8e27",
    "runtime_snapshot_sha256": (
        "8783e948c98eac9c8bf0c9ff8fa9f2c2dd6ff45174b53c181e078d08db081b2a"
    ),
    "terminal_reply_sha256": (
        "509077aba9ca82df2b348f2c0e09ed814ce38ccf7d8d3e89d49524b39736a26b"
    ),
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_nested_evidence(value: dict[str, Any]) -> None:
    dependency = value.get("dependency_stack")
    if dependency != live_owner_dependency_payload(ROOT):
        raise ValueError("Phase-2 dependency stack evidence changed")

    abi = value.get("abi_probe")
    expected_abi = {
        "schema_version": 1,
        "domain": "redco-stage-d2-qa-disposable-abi-v1",
        "python": "3.12.3",
        "split_engine_sampling": True,
        "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
        "sampling_field_count": 12,
        "episode_trace_model_call_v2_validated": True,
        "v2_provenance_record_count": 2,
        "prepared_observer_methods": list(EXPECTED_OBSERVER_METHODS),
        "controller_watchdog_interface": True,
        "module_bindings": list(EXPECTED_MODULE_BINDINGS),
    }
    if (
        not isinstance(abi, dict)
        or set(abi) != ABI_FIELDS
        or any(
            not isinstance(binding, dict)
            or set(binding) != MODULE_BINDING_FIELDS
            for binding in cast(list[object], abi.get("module_bindings", []))
        )
        or abi != expected_abi
    ):
        raise ValueError("Phase-2 ABI evidence changed")

    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or runtime != EXPECTED_RUNTIME:
        raise ValueError("Phase-2 runtime evidence changed")

    tests = value.get("tests")
    if (
        not isinstance(tests, dict)
        or set(tests) != TEST_FIELDS
        or tests
        != {
            "nodes": list(TEST_NODES),
            "collected": 3,
            "passed": 3,
            "failed": 0,
            "skipped": 0,
            "xfail": 0,
        }
    ):
        raise ValueError("Phase-2 test evidence changed")

    positive = value.get("positive")
    if (
        not isinstance(positive, dict)
        or set(positive) != POSITIVE_EVIDENCE_FIELDS
        or positive.get("schema_version") != 1
        or positive.get("domain") != "redco-stage-d2-qa-positive-evidence-v1"
        or positive.get("receipt_sha256") != REVIEWED_RECEIPT_SHA256
        or positive.get("report_sha256") != REVIEWED_REPORT_SHA256
        or any(positive.get(name) != digest for name, digest in EXPECTED_POSITIVE_HASHES.items())
        or positive.get("receipt_bytes") != 552
        or positive.get("reward") != 1.0
        or positive.get("loopback_fixture_calls") != 2
        or positive.get("qa_receipt_record_count") != 1
        or positive.get("qa_barrier_record_count") != 1
        or any(
            positive.get(name) != 0
            for name in (
                "provider_dispatches",
                "generated_tokens",
                "judge_calls",
                "candidate_records",
                "scientific_execution_records",
                "qasper_rows_read",
                "launch_dataset_reads",
                "campaign_recovery_callback_calls",
                "duplicate_run_eval_calls",
            )
        )
        or any(
            positive.get(name) is not True
            for name in (
                "sampling_directions_drained",
                "checkout_outputs_unchanged",
                "launch_claims_unchanged",
            )
        )
    ):
        raise ValueError("Phase-2 positive QA evidence changed")

    external = value.get("external_activity")
    if (
        not isinstance(external, dict)
        or set(external) != EXTERNAL_ACTIVITY_FIELDS
        or external
        != {
            "prime_calls": 0,
            "provider_calls": 0,
            "model_calls": 0,
            "gpu_calls": 0,
            "wallet_calls": 0,
            "external_network_calls": 0,
            "loopback_fixture_calls": 2,
            "qasper_rows_read": 0,
            "parquet_access": False,
            "ordinal_181_accessed": False,
            "launch_dataset_reads": 0,
            "external_prime_rl_imported": False,
            "external_prime_rl_modified": False,
        }
    ):
        raise ValueError("Phase-2 external-activity evidence changed")


def _load_evidence(path: Path) -> tuple[bytes, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("verification evidence must remain outside the checkout")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("verification evidence must be a regular file")
    raw = resolved.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != EVIDENCE_FIELDS
        or canonical_json(value) != raw
        or _sha256(raw) != REVIEWED_EVIDENCE_SHA256
        or value.get("schema_version") != 1
        or value.get("domain")
        != "redco-stage-d2-qa-disposable-overlay-evidence-v1"
        or value.get("parent_commit") != PARENT_COMMIT
        or value.get("parent_tree") != PARENT_TREE
        or value.get("dependency_authentication_runs") != 2
    ):
        raise ValueError("Phase-2 disposable-overlay evidence is invalid")
    _validate_nested_evidence(cast(dict[str, Any], value))
    return raw, cast(dict[str, Any], value)


def build_audit(evidence_path: Path) -> bytes:
    if _git_value("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ValueError("Phase-2 v2 audit must remain based on checkpoint 1ae6822")
    if _git_value("rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("Phase-2 checkpoint tree changed")
    v1_bytes = (ROOT / V1_AUDIT_RELATIVE).read_bytes()
    if _sha256(v1_bytes) != V1_AUDIT_SHA256:
        raise ValueError("the reviewed Phase-2 v1 audit changed")
    evidence_bytes, evidence = _load_evidence(evidence_path)
    file_bindings = {
        relative: _sha256((ROOT / relative).read_bytes()) for relative in BOUND_PATHS
    }
    return cast(
        bytes,
        canonical_json(
            {
                "schema_version": 2,
                "domain": "redco-stage-d2-qa-localization-audit-v2",
                "state": "non_authorizing_cpu_infrastructure_evidence",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": [*BOUND_PATHS, AUDIT_RELATIVE.as_posix()],
                "file_bindings": file_bindings,
                "self_hash": "excluded_to_avoid_circular_binding",
                "prior_audit": {
                    "path": V1_AUDIT_RELATIVE.as_posix(),
                    "sha256": V1_AUDIT_SHA256,
                    "unchanged": True,
                },
                "verification_evidence": {
                    "sha256": _sha256(evidence_bytes),
                    "payload": evidence,
                },
                "classification": {
                    "infrastructure_evidence_only": True,
                    "support_denominator_evidence": False,
                    "launch_bundle_created_or_modified": False,
                },
                "authorization": {
                    "candidate_selection_authorized": False,
                    "launch_authorized": False,
                    "phase_2_live_authorized": False,
                    "provider_calls_authorized": False,
                    "science_authorized": False,
                    "support_launch_authorized": False,
                },
            }
        ),
    )


def _publish(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("xb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(pending, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    target = output_root / AUDIT_RELATIVE
    expected = build_audit(args.evidence)
    if args.check_only:
        if target.read_bytes() != expected:
            raise ValueError("Phase-2 v2 audit differs from deterministic expected bytes")
    else:
        _publish(target, expected)
    print(
        json.dumps(
            {
                "bytes": len(expected),
                "path": AUDIT_RELATIVE.as_posix(),
                "sha256": _sha256(expected),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
