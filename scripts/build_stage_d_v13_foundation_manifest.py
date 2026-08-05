"""Build/check the candidate-null Foundation F tree witness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_foundation import (
    APPROVAL_ANCHOR_RELATIVE,
    FOUNDATION_MANIFEST_RELATIVE,
    SOURCE_PROVENANCE_RELATIVE,
    build_foundation_manifest,
    build_integrity_anchor,
    build_source_provenance,
    validate_foundation_manifest,
)
from redco.analysis.stage_d_v13_source_phase_a import (
    PHASE_A_OUTPUTS,
    foundation_envelope,
)
from redco.analysis.stage_d_v13_source_phase_a_trust import authenticate_external_anchor

ROOT = Path(__file__).resolve().parents[1]


def _replace_anchor(value: Any, approval: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if value.get("anchor_path") == APPROVAL_ANCHOR_RELATIVE:
            return dict(approval)
        return {key: _replace_anchor(item, approval) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_anchor(item, approval) for item in value]
    return value


def _refresh_candidate_null_phase_a_envelopes(root: Path) -> None:
    """Refresh only F envelope/status bindings when PyArrow 25 is unavailable."""

    approval = authenticate_external_anchor(root)
    paths = {relative: root / relative for relative in PHASE_A_OUTPUTS}
    values = {
        relative: json.loads(path.read_bytes()) for relative, path in paths.items()
    }
    for relative, value in values.items():
        if not isinstance(value, dict):
            raise ValueError(f"Phase-A artifact is not an object: {relative}")
        values[relative] = foundation_envelope(_replace_anchor(value, approval))

    status_relative = PHASE_A_OUTPUTS[4]
    status = values[status_relative]
    from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_A_STATUS_SIGNATURE
    from tests.test_stage_d_v13_source_phase_a import TEST_NODE_IDS

    expected_status = {
        "passed": len(TEST_NODE_IDS),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    status = foundation_envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-source-phase-a-status-v2",
            "status": expected_status,
            "node_ids": list(TEST_NODE_IDS),
            "status_signature": PHASE_A_STATUS_SIGNATURE,
            "capture": "independent_cpu_suite_status_signature",
            "status_semantics": "expected_collection_signature_not_observed",
            "execution": {
                "state": "blocked_missing_pinned_pyarrow_25",
                "observed": None,
            },
            "refresh_basis": "foundation_only_envelope_refresh; source_recomputation_blocked",
        }
    )
    values[status_relative] = status

    config_relative, audit_relative, cpu_relative, manifest_relative, _ = PHASE_A_OUTPUTS
    config = values[config_relative]
    audit = values[audit_relative]
    cpu = values[cpu_relative]
    manifest = values[manifest_relative]
    config_bytes = canonical_json_bytes(config)
    audit["config"] = {"path": config_relative, "sha256": sha256_bytes(config_bytes)}
    audit["refresh_basis"] = {
        "kind": "foundation_only_envelope_refresh",
        "source_recomputation": "blocked_missing_pinned_pyarrow_25",
        "prior_authenticated_prefix_evidence_retained": True,
    }
    cpu_suite = cpu["suite"]
    if not isinstance(cpu_suite, dict):
        raise ValueError("Phase-A CPU suite is not an object")
    cpu_suite["node_ids"] = list(TEST_NODE_IDS)
    cpu_suite["node_count"] = len(TEST_NODE_IDS)
    cpu_suite["node_list_sha256"] = sha256_bytes(canonical_json_bytes(list(TEST_NODE_IDS)))
    cpu_suite["expected"] = expected_status
    cpu_suite["observed"] = None
    cpu_suite["test_source_sha256"] = sha256_bytes(
        (root / "tests/test_stage_d_v13_source_phase_a.py").read_bytes()
    )
    cpu_suite["verification"] = {
        "collection_reproduced": True,
        "status_signature": PHASE_A_STATUS_SIGNATURE,
        "status_capture_path": status_relative,
        "status_capture_sha256": sha256_bytes(canonical_json_bytes(status)),
        "independent_status_capture": True,
        "source_recomputation": "blocked_missing_pinned_pyarrow_25",
        "execution_state": "blocked_missing_pinned_pyarrow_25",
    }
    cpu["refresh_basis"] = "foundation_only_envelope_refresh; source_recomputation_blocked"
    phase_hashes = {
        relative: sha256_bytes(canonical_json_bytes(values[relative]))
        for relative in (config_relative, audit_relative, cpu_relative, status_relative)
    }
    manifest["phase_a_artifacts"] = phase_hashes
    manifest["builder_sources"] = {
        relative: sha256_bytes((root / relative).read_bytes())
        for relative in manifest.get("builder_sources", {})
    }
    manifest["refresh_basis"] = {
        "kind": "foundation_only_envelope_refresh",
        "source_recomputation": "blocked_missing_pinned_pyarrow_25",
        "prior_authenticated_prefix_evidence_retained": True,
    }
    values[config_relative] = config
    values[audit_relative] = audit
    values[cpu_relative] = cpu
    values[manifest_relative] = manifest
    for relative, value in values.items():
        path = paths[relative]
        path.write_bytes(canonical_json_bytes(value))


def _write_generated_input(root: Path) -> None:
    path = root / SOURCE_PROVENANCE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(build_source_provenance())
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def _write_integrity_anchor(root: Path) -> None:
    path = root / APPROVAL_ANCHOR_RELATIVE
    data = canonical_json_bytes(build_integrity_anchor(root))
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _require_exact_bytes(path: Path, expected: bytes, *, label: str) -> None:
    """Compare a generated input without creating or replacing anything."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise ValueError(f"{label} bytes differ from the in-memory reconstruction")


def build(*, root: Path = ROOT) -> str:
    _write_generated_input(root)
    _write_integrity_anchor(root)
    _refresh_candidate_null_phase_a_envelopes(root)
    data = canonical_json_bytes(build_foundation_manifest(root))
    destination = root / FOUNDATION_MANIFEST_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    if destination.read_bytes() != data:
        raise AssertionError("Foundation F manifest write was not byte exact")
    return cast(str, sha256_bytes(data))


def check_only(*, root: Path = ROOT) -> None:
    # Check-only is deliberately a read-only authentication boundary.  Build
    # the expected generated inputs in memory and compare their exact bytes;
    # never call a writer, create a parent directory, or repair a witness.
    _require_exact_bytes(
        root / SOURCE_PROVENANCE_RELATIVE,
        canonical_json_bytes(build_source_provenance()),
        label="QASPER source provenance",
    )
    _require_exact_bytes(
        root / APPROVAL_ANCHOR_RELATIVE,
        canonical_json_bytes(build_integrity_anchor(root)),
        label="Phase-A approval anchor",
    )
    path = root / FOUNDATION_MANIFEST_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(FOUNDATION_MANIFEST_RELATIVE)
    raw = path.read_bytes()
    import json

    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("Foundation F manifest is not canonical")
    validate_foundation_manifest(root, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.check_only:
        check_only(root=args.root)
    else:
        print(build(root=args.root))


if __name__ == "__main__":
    main()
