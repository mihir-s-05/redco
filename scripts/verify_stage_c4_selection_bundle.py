"""Verify and summarize a compact terminal Stage-C4 selection bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _load_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    verify_signed_payload(payload)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, manifest: Path) -> int:
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = root / relative
        if _sha256(path) != expected:
            raise ValueError(f"manifest mismatch for {relative}")
        checked += 1
    return checked


def _verify_scorer_pair(
    bundle_root: Path,
    directory: Path,
    stem: str,
) -> dict[str, Any]:
    score_path = directory / f"{stem}-scores.json"
    verified_path = directory / f"{stem}-scores.verified.json"
    score = _load_signed(score_path)
    sentinel = _load_signed(verified_path)
    if sentinel["status"] != "verified" or sentinel["child_returncode"] != 0:
        raise ValueError(f"{verified_path} does not record a zero-exit verification")
    recorded_output = bundle_root / sentinel["output_path"]
    if recorded_output.resolve() != score_path.resolve():
        raise ValueError(f"{verified_path} points to the wrong output")
    if sentinel["output_file_sha256"] != _sha256(score_path):
        raise ValueError(f"{verified_path} has the wrong output file hash")
    if sentinel["output_signed_payload_sha256"] != score["signed_payload_sha256"]:
        raise ValueError(f"{verified_path} has the wrong signed payload hash")
    return score


def verify(bundle_root: Path) -> dict[str, Any]:
    stage_root = bundle_root / "runs/stage-c4"
    selection_root = stage_root / "warmstart-selection-v2"
    lifecycle_root = stage_root / "scorer-lifecycle-v2"
    sft_root = stage_root / "warmstart-sft-v2"
    if not (selection_root / "SELECTION_TERMINAL_FAILURE").is_file():
        raise ValueError("terminal selection marker is missing")
    if not (lifecycle_root / "LIFECYCLE_GATE_PASSED").is_file():
        raise ValueError("lifecycle pass marker is missing")

    manifest_counts = {
        "selection": _verify_manifest(
            bundle_root,
            selection_root / "sha256-manifest.txt",
        ),
        "lifecycle": _verify_manifest(
            bundle_root,
            lifecycle_root / "sha256-manifest.txt",
        ),
    }
    _verify_scorer_pair(bundle_root, lifecycle_root, "action")
    _verify_scorer_pair(bundle_root, lifecycle_root, "root")

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
                "status": report["status"],
                "mean_digit_5_mass_t2": measurements["mean_digit_5_mass_t2"],
                "valid_route_sequence_mass_t2": measurements["valid_route_sequence_mass_t2"],
                "delta_route_sequence_mass_t2": measurements["route_sequence_probabilities_t2"][
                    "delta"
                ],
                "gamma_route_sequence_mass_t2": measurements["route_sequence_probabilities_t2"][
                    "gamma"
                ],
                "normalized_root_entropy_nats": report["model_factorization"][
                    "normalized_root_entropy_nats"
                ],
                "expected_target_informative_groups": measurements[
                    "expected_target_informative_groups_per_sliced_step"
                ],
                "root_group_informative_probability": measurements[
                    "root_group_informative_probability"
                ],
                "redundant_group_informative_probability": measurements[
                    "redundant_broadcast_group_informative_probability_lower"
                ],
                "route_digit_joint_tv": report["model_factorization"][
                    "route_digit_joint_total_variation"
                ],
                "route_digit_mutual_information_nats": report["model_factorization"][
                    "route_digit_mutual_information_nats"
                ],
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
            "analysis": "stage-c4-warmstart-selection-v2-terminal-verification",
            "status": "verified-terminal-no-selection",
            "manifest_file_counts": manifest_counts,
            "lifecycle_scorers_verified": 2,
            "candidate_scorer_pairs_verified": 16,
            "candidate_reports_verified": 16,
            "selection_signed_payload_sha256": selection["signed_payload_sha256"],
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
