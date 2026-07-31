"""Report the terminal Stage D v4.6 retained-adapter continuation."""

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

RETENTION_MEMBER = (
    "runs/stage-d0/scaffold-support-v4-6/retention-ledger.json"
)
HEALTH_MEMBER = "runs/stage-d0/scaffold-support-v4-6/runtime-health.json"
FIXTURE_LOG_MEMBER = (
    "runs/stage-d0/scaffold-support-v4-6/"
    "selected-fixture-control.log"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_member(bundle: tarfile.TarFile, name: str) -> bytes:
    extracted = bundle.extractfile(name)
    if extracted is None:
        raise ValueError(f"evidence member is missing: {name}")
    return extracted.read()


def _read_signed_json(
    bundle: tarfile.TarFile,
    name: str,
) -> dict[str, Any]:
    payload = json.loads(_read_member(bundle, name))
    verify_signed_payload(payload)
    return payload


def report(
    *,
    evidence: Path,
    protocol: Path,
    audit: Path,
    transfer_repair: Path,
    fixture_dataset: Path,
    taskset_source: Path,
) -> dict[str, Any]:
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    verify_signed_payload(audit_payload)
    transfer_payload = json.loads(
        transfer_repair.read_text(encoding="utf-8")
    )
    fixture_rows = [
        json.loads(line)
        for line in fixture_dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    taskset_text = taskset_source.read_text(encoding="utf-8")

    with tarfile.open(evidence, mode="r:gz") as bundle:
        names = {
            member.name
            for member in bundle.getmembers()
            if member.isfile()
        }
        retention = _read_signed_json(bundle, RETENTION_MEMBER)
        health = _read_signed_json(bundle, HEALTH_MEMBER)
        fixture_log = _read_member(
            bundle,
            FIXTURE_LOG_MEMBER,
        ).decode("utf-8", errors="replace")

    fixture_outputs = {
        name
        for name in names
        if name.startswith(
            "runs/stage-d0/scaffold-support-v4-6/selected-fixture/"
        )
    }
    power_outputs = {
        name
        for name in names
        if name.startswith(
            "runs/stage-d0/scaffold-support-v4-6/power-audit/"
        )
    }
    checks = {
        "protocol_frozen_and_audit_passed": (
            protocol_payload["status"]
            == "frozen_before_any_v4_6_model_call"
            and audit_payload["passes"]
            and audit_payload["protocol_sha256"] == _sha256(protocol)
        ),
        "retained_archive_identity_exact": (
            retention["archive_sha256"]
            == protocol_payload["fixed_initialization"][
                "adapter_archive_sha256"
            ]
        ),
        "isolated_canonical_retention_passed": (
            retention["passes"]
            and retention["action_payloads_byte_identical"]
            and retention["root_payloads_byte_identical"]
            and len(retention["invocations"]) == 2
        ),
        "merged_vllm_health_passed": (
            health["passes"] and all(health["checks"].values())
        ),
        "fixture_schema_failure_exact": (
            "KeyError: 'answer_type'" in fixture_log
            and "answer_type" not in fixture_rows[0]
            and 'row["answer_type"]' in taskset_text
        ),
        "no_fixture_output_observed": not fixture_outputs,
        "no_power_output_observed": not power_outputs,
        "bounded_repair_was_exhausted": transfer_payload[
            "redeployment_rule"
        ]["no_further_environment_repair_or_redeployment"],
    }
    attempt_cost = 0.1484 + 0.1748
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-v4-6-terminal",
            "status": "terminal_before_fixture_model_call_and_power_audit",
            "controlling_protocol": protocol.as_posix(),
            "controlling_protocol_sha256": _sha256(protocol),
            "controlling_audit": audit.as_posix(),
            "controlling_audit_sha256": _sha256(audit),
            "transfer_repair": transfer_repair.as_posix(),
            "transfer_repair_sha256": _sha256(transfer_repair),
            "evidence": {
                "archive": evidence.as_posix(),
                "archive_sha256": _sha256(evidence),
                "archive_byte_length": evidence.stat().st_size,
                "members": sorted(names),
            },
            "validated_results": {
                "retention": {
                    "passes": retention["passes"],
                    "signed_payload_sha256": retention[
                        "signed_payload_sha256"
                    ],
                    "action_payloads_byte_identical": retention[
                        "action_payloads_byte_identical"
                    ],
                    "root_payloads_byte_identical": retention[
                        "root_payloads_byte_identical"
                    ],
                },
                "merged_vllm_health": {
                    "passes": health["passes"],
                    "signed_payload_sha256": health[
                        "signed_payload_sha256"
                    ],
                    "checks": health["checks"],
                },
            },
            "terminal_failure": {
                "stage": "selected_fixture_task_loading",
                "failure": (
                    "The frozen fixture row lacks answer_type, while the "
                    "shared v2 task loader requires row['answer_type']."
                ),
                "outcome_independent": True,
                "fixture_model_calls": 0,
                "fixture_outputs": 0,
                "power_audit_model_calls": 0,
                "scientific_arm_outcomes": 0,
                "interpretation": (
                    "This does not invalidate the exact adapter-retention or "
                    "merged-vLLM health passes. It leaves target eligibility "
                    "and Stage D power unmeasured."
                ),
            },
            "resources": {
                "attempt_1_cost_usd": 0.1484,
                "attempt_2_cost_usd": 0.1748,
                "v4_6_total_cost_usd": attempt_cost,
                "wallet_before_v4_6_usd": 46.3976,
                "wallet_after_v4_6_usd": 46.0744,
                "wallet_delta_usd": 46.3976 - 46.0744,
                "active_pods_after_termination": 0,
                "persistent_disks_after_termination": 0,
            },
            "disposition": (
                "Close v4.6. Do not restart or redeploy it. Any successor "
                "must be separately frozen and must repair the fixture schema "
                "with an end-to-end task-loading preflight before paid work."
            ),
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v4-6.json"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "reports/"
            "stage-d0-scaffold-support-preregistration-audit-v4-6.json"
        ),
    )
    parser.add_argument(
        "--transfer-repair",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-transfer-repair-v4-6-2.json"
        ),
    )
    parser.add_argument(
        "--fixture-dataset",
        type=Path,
        default=Path(
            "datasets/stage-d/evidence-selection-fixture-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--taskset-source",
        type=Path,
        default=Path(
            "environments/redco_evidence_selection_v2/"
            "redco_evidence_selection_v2/taskset.py"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = report(
        evidence=args.evidence,
        protocol=args.protocol,
        audit=args.audit,
        transfer_repair=args.transfer_repair,
        fixture_dataset=args.fixture_dataset,
        taskset_source=args.taskset_source,
    )
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
