"""Verify and summarize the compact terminal Stage-C4 selection-v3 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_stage_c4_selection_bundle import (
    _load_signed,
    _verify_manifest,
    _verify_scorer_pair,
)

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def verify(bundle_root: Path) -> dict:
    stage_root = bundle_root / "runs/stage-c4"
    selection_root = stage_root / "warmstart-selection-v3"
    sft_root = stage_root / "warmstart-sft-v3"
    if not (selection_root / "SELECTION_TERMINAL_FAILURE").is_file():
        raise ValueError("terminal selection marker is missing")

    manifest_count = _verify_manifest(
        bundle_root,
        selection_root / "sha256-manifest.txt",
    )
    alignment = _load_signed(selection_root / "renderer-alignment.json")
    if alignment["status"] != "passed" or not all(alignment["checks"].values()):
        raise ValueError("renderer alignment did not pass")
    live_prereg = _load_signed(stage_root / "v3-preregistration-audit-live.json")
    if not live_prereg["passed"]:
        raise ValueError("live preregistration audit did not pass")
    _load_signed(selection_root / "base-merge-manifest.json")

    curve = []
    report_signatures: dict[str, str] = {}
    for step in range(1, 17):
        candidate = selection_root / "candidates" / f"step_{step}"
        _verify_scorer_pair(bundle_root, candidate, "action")
        _verify_scorer_pair(bundle_root, candidate, "root")
        report = _load_signed(candidate / "report.json")
        if report["step"] != step or report["status"] != "failed":
            raise ValueError(f"candidate {step} has an unexpected disposition")
        report_signatures[str(step)] = report["signed_payload_sha256"]
        measurements = report["campaign_power"]["measurements"]
        curve.append(
            {
                "step": step,
                "mean_digit_5_mass_t2": measurements["mean_digit_5_mass_t2"],
                "valid_route_sequence_mass_t2": measurements[
                    "valid_route_sequence_mass_t2"
                ],
                "route_sequence_probabilities_t2": measurements[
                    "route_sequence_probabilities_t2"
                ],
                "normalized_root_entropy_nats": report["model_factorization"][
                    "normalized_root_entropy_nats"
                ],
                "expected_target_informative_groups": measurements[
                    "expected_target_informative_groups_per_sliced_step"
                ],
                "redundant_group_informative_probability": measurements[
                    "redundant_broadcast_group_informative_probability_lower"
                ],
                "route_digit_joint_tv": report["model_factorization"][
                    "route_digit_joint_total_variation"
                ],
                "route_digit_mutual_information_nats": report[
                    "model_factorization"
                ]["route_digit_mutual_information_nats"],
            }
        )

    selection = _load_signed(selection_root / "selection.json")
    if (
        selection["status"] != "failed"
        or selection["selected_step"] is not None
        or selection["evaluated_steps"] != list(range(1, 17))
        or selection["candidate_signed_payloads"] != report_signatures
    ):
        raise ValueError("aggregate selection does not match candidate reports")

    loss_by_step: dict[int, float] = {}
    nan_by_step: dict[int, int] = {}
    for line in (sft_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "loss/mean" in row:
            loss_by_step[int(row["step"])] = float(row["loss/mean"])
            nan_by_step[int(row["step"])] = int(row["loss/nan_count"])
    if set(loss_by_step) != set(range(1, 17)) or any(nan_by_step.values()):
        raise ValueError("SFT metrics do not contain 16 finite loss steps")

    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c4-warmstart-selection-v3-terminal-verification",
            "status": "verified-terminal-no-selection",
            "selection_manifest_files_verified": manifest_count,
            "renderer_alignment_signature": alignment["signed_payload_sha256"],
            "live_preregistration_signature": live_prereg[
                "signed_payload_sha256"
            ],
            "candidate_scorer_pairs_verified": 16,
            "candidate_reports_verified": 16,
            "selection_signed_payload_sha256": selection[
                "signed_payload_sha256"
            ],
            "sft": {
                "steps": 16,
                "initial_loss": loss_by_step[1],
                "final_loss": loss_by_step[16],
                "nan_steps": sum(nan_by_step.values()),
            },
            "curve": curve,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.bundle_root)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "signature": result["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
