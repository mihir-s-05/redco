"""Canonical owners for the frozen Stage-C4 V2-V4 bundle verifiers."""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)

JsonObject = dict[str, Any]
Verifier = Callable[[Path], JsonObject]


class FactorizedSpec(NamedTuple):
    version: Literal[3, 4]
    candidate_steps: tuple[int, ...]
    sft_step_count: int


V3_SPEC = FactorizedSpec(3, tuple(range(1, 17)), 16)
V4_SPEC = FactorizedSpec(4, tuple(range(2, 33, 2)), 32)


def load_signed_json(path: Path) -> JsonObject:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object")
    payload = cast(JsonObject, loaded)
    verify_signed_payload(payload)
    return payload


def _manifest(root: Path, manifest: Path) -> int:
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch for {relative}")
        checked += 1
    return checked


def verify_scorer_pair(bundle_root: Path, directory: Path, stem: str) -> JsonObject:
    score_path = directory / f"{stem}-scores.json"
    verified_path = directory / f"{stem}-scores.verified.json"
    score = load_signed_json(score_path)
    sentinel = load_signed_json(verified_path)
    if sentinel["status"] != "verified" or sentinel["child_returncode"] != 0:
        raise ValueError(f"{verified_path} does not record a zero-exit verification")
    if (bundle_root / sentinel["output_path"]).resolve() != score_path.resolve():
        raise ValueError(f"{verified_path} points to the wrong output")
    score_digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
    if sentinel["output_file_sha256"] != score_digest:
        raise ValueError(f"{verified_path} has the wrong output file hash")
    if sentinel["output_signed_payload_sha256"] != score["signed_payload_sha256"]:
        raise ValueError(f"{verified_path} has the wrong signed payload hash")
    return score


def _curve_row(step: int, report: JsonObject, factorized: bool) -> JsonObject:
    measurements = report["campaign_power"]["measurements"]
    factorization = report["model_factorization"]
    informative = measurements["expected_target_informative_groups_per_sliced_step"]
    redundant = measurements["redundant_broadcast_group_informative_probability_lower"]
    mutual_information = factorization["route_digit_mutual_information_nats"]
    row = {
        "step": step,
        "mean_digit_5_mass_t2": measurements["mean_digit_5_mass_t2"],
        "valid_route_sequence_mass_t2": measurements["valid_route_sequence_mass_t2"],
        "normalized_root_entropy_nats": factorization["normalized_root_entropy_nats"],
        "expected_target_informative_groups": informative,
        "redundant_group_informative_probability": redundant,
        "route_digit_joint_tv": factorization["route_digit_joint_total_variation"],
        "route_digit_mutual_information_nats": mutual_information,
    }
    probabilities = measurements["route_sequence_probabilities_t2"]
    if factorized:
        row["route_sequence_probabilities_t2"] = probabilities
    else:
        row["status"] = report["status"]
        row["delta_route_sequence_mass_t2"] = probabilities["delta"]
        row["gamma_route_sequence_mass_t2"] = probabilities["gamma"]
        row["root_group_informative_probability"] = measurements[
            "root_group_informative_probability"
        ]
    return row


def _terminal_selection(
    bundle_root: Path,
    selection_root: Path,
    sft_root: Path,
    candidate_steps: Sequence[int],
    step_count: int,
    factorized: bool,
    version: Literal[2, 3, 4],
    extras: JsonObject,
) -> JsonObject:
    curve: list[JsonObject] = []
    signatures: dict[str, str] = {}
    for step in candidate_steps:
        candidate = selection_root / "candidates" / f"step_{step}"
        verify_scorer_pair(bundle_root, candidate, "action")
        verify_scorer_pair(bundle_root, candidate, "root")
        report = load_signed_json(candidate / "report.json")
        if report["step"] != step or report["status"] != "failed":
            raise ValueError(f"candidate {step} has an unexpected disposition")
        signatures[str(step)] = report["signed_payload_sha256"]
        curve.append(_curve_row(step, report, factorized))
    selection = load_signed_json(selection_root / "selection.json")
    if (
        selection["status"] != "failed"
        or selection["selected_step"] is not None
        or selection["evaluated_steps"] != list(candidate_steps)
        or selection["candidate_signed_payloads"] != signatures
    ):
        raise ValueError("aggregate selection does not match candidate reports")
    loss_by_step: dict[int, float] = {}
    nan_by_step: dict[int, int] = {}
    for line in (sft_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "loss/mean" in row:
            loss_by_step[int(row["step"])] = float(row["loss/mean"])
            nan_by_step[int(row["step"])] = int(row["loss/nan_count"])
    if set(loss_by_step) != set(range(1, step_count + 1)) or any(nan_by_step.values()):
        raise ValueError(f"SFT metrics do not contain {step_count} finite loss steps")
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": f"stage-c4-warmstart-selection-v{version}-terminal-verification",
            "status": "verified-terminal-no-selection",
            **extras,
            "candidate_scorer_pairs_verified": len(candidate_steps),
            "candidate_reports_verified": len(candidate_steps),
            "selection_signed_payload_sha256": selection["signed_payload_sha256"],
            "sft": {
                "steps": step_count,
                "initial_loss": loss_by_step[1],
                "final_loss": loss_by_step[step_count],
                "nan_steps": sum(nan_by_step.values()),
            },
            "curve": curve,
        }
    )


def verify_v2_bundle(bundle_root: Path) -> JsonObject:
    stage_root = bundle_root / "runs/stage-c4"
    selection_root = stage_root / "warmstart-selection-v2"
    lifecycle_root = stage_root / "scorer-lifecycle-v2"
    if not (selection_root / "SELECTION_TERMINAL_FAILURE").is_file():
        raise ValueError("terminal selection marker is missing")
    if not (lifecycle_root / "LIFECYCLE_GATE_PASSED").is_file():
        raise ValueError("lifecycle pass marker is missing")
    manifest_counts = {
        "selection": _manifest(bundle_root, selection_root / "sha256-manifest.txt"),
        "lifecycle": _manifest(bundle_root, lifecycle_root / "sha256-manifest.txt"),
    }
    verify_scorer_pair(bundle_root, lifecycle_root, "action")
    verify_scorer_pair(bundle_root, lifecycle_root, "root")
    return _terminal_selection(
        bundle_root,
        selection_root,
        stage_root / "warmstart-sft-v2",
        range(1, 17),
        16,
        False,
        2,
        {"manifest_file_counts": manifest_counts, "lifecycle_scorers_verified": 2},
    )


def verify_factorized(bundle_root: Path, *, spec: FactorizedSpec) -> JsonObject:
    stage_root = bundle_root / "runs/stage-c4"
    selection_root = stage_root / f"warmstart-selection-v{spec.version}"
    if not (selection_root / "SELECTION_TERMINAL_FAILURE").is_file():
        raise ValueError("terminal selection marker is missing")
    manifest_count = _manifest(bundle_root, selection_root / "sha256-manifest.txt")
    alignment = load_signed_json(selection_root / "renderer-alignment.json")
    if alignment["status"] != "passed" or not all(alignment["checks"].values()):
        raise ValueError("renderer alignment did not pass")
    if spec.version == 3:
        preregistration_path = stage_root / "v3-preregistration-audit-live.json"
        signature_key = "live_preregistration_signature"
        failure = "live preregistration audit did not pass"
    else:
        preregistration_path = bundle_root / (
            "reports/stage-c4-warmstart-selection-v4-preregistration-audit-2026-07-29.json"
        )
        signature_key = "preregistration_audit_signature"
        failure = "frozen preregistration audit did not pass"
    preregistration = load_signed_json(preregistration_path)
    if not preregistration["passed"]:
        raise ValueError(failure)
    load_signed_json(selection_root / "base-merge-manifest.json")
    return _terminal_selection(
        bundle_root,
        selection_root,
        stage_root / f"warmstart-sft-v{spec.version}",
        spec.candidate_steps,
        spec.sft_step_count,
        True,
        spec.version,
        {
            "selection_manifest_files_verified": manifest_count,
            "renderer_alignment_signature": alignment["signed_payload_sha256"],
            signature_key: preregistration["signed_payload_sha256"],
        },
    )


def run_verification_cli(*, description: str | None, verifier: Verifier) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verifier(args.bundle_root)
    atomic_write_json(args.output, result)
    summary = {"status": result["status"], "signature": result["signed_payload_sha256"]}
    print(json.dumps(summary, sort_keys=True))
