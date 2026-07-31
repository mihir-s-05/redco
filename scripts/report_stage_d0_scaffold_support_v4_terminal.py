"""Record the terminal Stage D0 scaffold-support v4.5 attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _models(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(model["name"]): model for model in payload["models"]}


def _reload_deltas(payload: dict[str, Any]) -> dict[str, Any]:
    models = _models(payload)
    original = models["original"]["temperatures"]
    reloaded = models["reloaded"]["temperatures"]
    rows = []
    greedy_equal = True
    for temperature in ("1.0", "2.0"):
        for left, right in zip(
            original[temperature],
            reloaded[temperature],
            strict=True,
        ):
            if left["case_id"] != right["case_id"]:
                raise ValueError("reload score case order differs")
            greedy_equal &= (
                int(left["greedy_token_id"]) == int(right["greedy_token_id"])
            )
            for action, left_probability in left[
                "action_probabilities"
            ].items():
                rows.append(
                    {
                        "temperature": temperature,
                        "case_id": left["case_id"],
                        "action": action,
                        "probability_delta": abs(
                            float(left_probability)
                            - float(right["action_probabilities"][action])
                        ),
                        "logprob_delta": abs(
                            float(left["action_logprobabilities"][action])
                            - float(
                                right["action_logprobabilities"][action]
                            )
                        ),
                    }
                )
    maximum_probability = max(rows, key=lambda row: row["probability_delta"])
    maximum_logprob = max(rows, key=lambda row: row["logprob_delta"])
    return {
        "all_greedy_tokens_equal": greedy_equal,
        "maximum_absolute_probability_delta": maximum_probability,
        "maximum_absolute_logprob_delta": maximum_logprob,
        "maximum_non_planted_probability_delta": max(
            float(row["probability_delta"])
            for row in rows
            if row["case_id"] != "planted_needle:gamma"
        ),
        "maximum_planted_needle_probability_delta": max(
            float(row["probability_delta"])
            for row in rows
            if row["case_id"] == "planted_needle:gamma"
        ),
    }


def report(
    *,
    run_root: Path,
    archive_manifest_path: Path,
    protocol_path: Path,
    amendment_path: Path,
    terminal_bundle: Path,
    pod_cost_usd: float,
    wallet_after_usd: float,
) -> dict[str, Any]:
    fewshot_path = run_root / "fewshot-support-audit.json"
    scores_path = run_root / "sft-reload-scores.json"
    sft_log_path = run_root / "sft-control.log"
    fewshot = json.loads(fewshot_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    archive_manifest = json.loads(
        archive_manifest_path.read_text(encoding="utf-8")
    )
    for payload in (fewshot, scores, archive_manifest):
        verify_signed_payload(payload)
    sft_log = sft_log_path.read_text(encoding="utf-8")
    observed_steps = sorted(
        {int(value) for value in re.findall(r"Step (\d+) \|", sft_log)}
    )
    artifact_hashes = {
        path.relative_to(run_root).as_posix(): _sha256(path)
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "analysis": "stage-d0-scaffold-support-v4-terminal",
        "status": "terminal_before_selected_fixture_and_power_audit",
        "controlling_protocol": protocol_path.as_posix(),
        "controlling_protocol_sha256": _sha256(protocol_path),
        "last_amendment": amendment_path.as_posix(),
        "last_amendment_sha256": _sha256(amendment_path),
        "frozen_cascade": {
            "fewshot_support": {
                "passes": fewshot["passes"],
                "rollouts": fewshot["rollouts"],
                "precursor_eligible": fewshot["precursor_eligible"],
                "required_precursor_eligible": fewshot[
                    "required_precursor_eligible"
                ],
                "median_child_calls": fewshot["median_child_calls"],
                "p95_child_calls": fewshot["p95_child_calls"],
                "parseable": sum(row["parseable"] for row in fewshot["rows"]),
                "verbatim": sum(row["verbatim"] for row in fewshot["rows"]),
                "signed_payload_sha256": fewshot["signed_payload_sha256"],
                "disposition": fewshot["cascade_disposition"],
            },
            "conditional_sft": {
                "classification": "shared synthetic scaffold-and-task SFT",
                "configured_steps": 8,
                "observed_steps": observed_steps,
                "finished": (
                    observed_steps == list(range(1, 9))
                    and "SFT trainer finished!" in sft_log
                ),
                "fixed_candidate": "step_8",
                "rerun_allowed": False,
            },
            "retention_test": {
                "archive_manifest": archive_manifest_path.as_posix(),
                "archive_manifest_sha256": _sha256(archive_manifest_path),
                "adapter_archive_sha256": archive_manifest["archive_sha256"],
                "adapter_model_sha256": archive_manifest["members"][
                    "adapter_model.safetensors"
                ]["sha256"],
                "archive_byte_identity_check": (
                    "passed in runner control flow before scoring"
                ),
                "score_payload_sha256": _sha256(scores_path),
                "score_payload_signature": scores[
                    "signed_payload_sha256"
                ],
                "score_payload_exact": False,
                "deltas": _reload_deltas(scores),
                "failure": (
                    "The original and re-extracted byte-identical adapters "
                    "occupied LoRA IDs 1 and 2 in one vLLM engine and did not "
                    "produce an exactly equal score payload."
                ),
            },
            "selected_fixture_model_calls": 0,
            "power_audit_model_calls": 0,
            "scientific_arm_outcomes": 0,
        },
        "interpretation": {
            "valid_findings": [
                (
                    "Few-shot prompting alone did not meet the frozen "
                    "scaffold-support gate."
                ),
                (
                    "The fixed eight-step SFT completed and its compressed "
                    "adapter is locally retained with a tensor-level manifest."
                ),
                (
                    "The co-resident multi-LoRA retention construction is not "
                    "a valid exact reload test on this pinned stack."
                ),
            ],
            "not_established": [
                "whether the retained step-8 adapter passes the exact selected fixture",
                "whether 58 of 64 independent papers are eligible and informative",
                "any Stage D learning comparison"
            ],
            "cause_scope": (
                "The byte-identical archive and probe-specific score pattern "
                "are consistent with a co-resident multi-LoRA slot/routing "
                "artifact. The exact internal vLLM root cause is not claimed."
            ),
        },
        "resource_cleanup": {
            "pod_id": "6bf9ef3733f648b8bb21adc98da62ad2",
            "pod_cost_usd": pod_cost_usd,
            "wallet_after_termination_usd": wallet_after_usd,
            "active_pods": 0,
            "persistent_disks": 0,
            "terminated": True,
        },
        "artifacts": {
            "run_root": run_root.as_posix(),
            "artifact_hashes": artifact_hashes,
            "terminal_bundle": terminal_bundle.as_posix(),
            "terminal_bundle_sha256": _sha256(terminal_bundle),
            "adapter_archive": (
                run_root / "selected-adapter.tar.gz"
            ).as_posix(),
        },
        "disposition": (
            "Close v4.5. A separately frozen successor may reuse only the "
            "recovered fixed step-8 adapter, may not rerun support or SFT, and "
            "must pass an isolated single-adapter retention micro-gate before "
            "any previously unobserved fixture or power-audit call."
        ),
    }
    return sign_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--terminal-bundle", type=Path, required=True)
    parser.add_argument("--pod-cost-usd", type=float, required=True)
    parser.add_argument("--wallet-after-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.output,
        report(
            run_root=args.run_root,
            archive_manifest_path=args.archive_manifest,
            protocol_path=args.protocol,
            amendment_path=args.amendment,
            terminal_bundle=args.terminal_bundle,
            pod_cost_usd=args.pod_cost_usd,
            wallet_after_usd=args.wallet_after_usd,
        ),
    )


if __name__ == "__main__":
    main()
