import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from redco.analysis.stage_c4_selection_bundle_verification import (
    load_signed_json,
    verify_scorer_pair,
)
from redco.integrations.signed_subprocess import sign_payload

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    2: "verify_stage_c4_selection_bundle.py",
    3: "verify_stage_c4_selection_v3_bundle.py",
    4: "verify_stage_c4_selection_v4_bundle.py",
}
DESCRIPTIONS = {
    2: "Verify and summarize a compact terminal Stage-C4 selection bundle.",
    3: "Verify and summarize the compact terminal Stage-C4 selection-v3 bundle.",
    4: "Verify and summarize the compact terminal Stage-C4 selection-v4 bundle.",
}
SIGNATURES = {
    2: "9f04b07ce20f9c05ce003744df4a0abb75810e56a80a5ce93e1cb15257224736",
    3: "a0f674e029b2cefb290d35d18980aec3517b9089f6b070820d6717c32c36aac2",
    4: "54dd4ddc4d4d5966451828b949c58a70adf0520d91e19756d17cd7a758cbe69c",
}
STEPS = (tuple(range(1, 17)),) * 2 + (tuple(range(2, 33, 2)),)


def _signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    signed = sign_payload(payload)
    path.write_text(json.dumps(signed, sort_keys=True), encoding="utf-8")
    return signed


def _scorer(bundle_root: Path, directory: Path, stem: str) -> None:
    score_path = directory / f"{stem}-scores.json"
    score = _signed(score_path, {"score": stem})
    _signed(
        directory / f"{stem}-scores.verified.json",
        {
            "status": "verified",
            "child_returncode": 0,
            "output_path": score_path.relative_to(bundle_root).as_posix(),
            "output_file_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
            "output_signed_payload_sha256": score["signed_payload_sha256"],
        },
    )


def _manifest(bundle_root: Path, manifest: Path, label: str) -> None:
    evidence = bundle_root / "evidence" / f"{label}.txt"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(f"{label}\n", encoding="utf-8")
    relative = evidence.relative_to(bundle_root).as_posix()
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest.write_text(f"{digest} *{relative}\n", encoding="utf-8")


def _report(step: int) -> dict[str, Any]:
    measurements = {
        "mean_digit_5_mass_t2": step / 100,
        "valid_route_sequence_mass_t2": step / 200,
        "route_sequence_probabilities_t2": {"delta": step / 300, "gamma": step / 400},
        "expected_target_informative_groups_per_sliced_step": step / 500,
        "root_group_informative_probability": step / 600,
        "redundant_broadcast_group_informative_probability_lower": step / 700,
    }
    return {
        "step": step,
        "status": "failed",
        "campaign_power": {"measurements": measurements},
        "model_factorization": {
            "normalized_root_entropy_nats": step / 800,
            "route_digit_joint_total_variation": step / 900,
            "route_digit_mutual_information_nats": step / 1000,
        },
    }


def _build(bundle_root: Path, version: int) -> tuple[int, ...]:
    stage = bundle_root / "runs/stage-c4"
    selection = stage / f"warmstart-selection-v{version}"
    selection.mkdir(parents=True)
    (selection / "SELECTION_TERMINAL_FAILURE").touch()
    _manifest(bundle_root, selection / "sha256-manifest.txt", f"selection-v{version}")
    if version == 2:
        lifecycle = stage / "scorer-lifecycle-v2"
        lifecycle.mkdir()
        (lifecycle / "LIFECYCLE_GATE_PASSED").touch()
        _manifest(bundle_root, lifecycle / "sha256-manifest.txt", "lifecycle-v2")
        _scorer(bundle_root, lifecycle, "action")
        _scorer(bundle_root, lifecycle, "root")
    else:
        _signed(
            selection / "renderer-alignment.json",
            {"status": "passed", "checks": {"renderer": True}},
        )
        _signed(selection / "base-merge-manifest.json", {"base": "ok"})
        preregistration = (
            stage / "v3-preregistration-audit-live.json"
            if version == 3
            else bundle_root / "reports/stage-c4-warmstart-selection-v4-"
            "preregistration-audit-2026-07-29.json"
        )
        _signed(preregistration, {"passed": True})
    signatures = {}
    steps = STEPS[version - 2]
    for step in steps:
        candidate = selection / "candidates" / f"step_{step}"
        _scorer(bundle_root, candidate, "action")
        _scorer(bundle_root, candidate, "root")
        report = _signed(candidate / "report.json", _report(step))
        signatures[str(step)] = report["signed_payload_sha256"]
    _signed(
        selection / "selection.json",
        {
            "status": "failed",
            "selected_step": None,
            "evaluated_steps": list(steps),
            "candidate_signed_payloads": signatures,
        },
    )
    step_count = 32 if version == 4 else 16
    metrics = stage / f"warmstart-sft-v{version}" / "metrics.jsonl"
    metrics.parent.mkdir()
    metrics.write_text(
        "".join(
            json.dumps(
                {"step": step, "loss/mean": float(step_count - step), "loss/nan_count": 0},
                sort_keys=True,
            )
            + "\n"
            for step in range(1, step_count + 1)
        ),
        encoding="utf-8",
    )
    return steps


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("version", [2, 3, 4])
def test_versioned_verifier_cli_preserves_terminal_protocol(
    tmp_path: Path,
    version: int,
) -> None:
    steps = _build(tmp_path, version)
    output = tmp_path / "terminal-verification.json"
    completed = _run(SCRIPTS[version], "--bundle-root", str(tmp_path), "--output", str(output))
    assert completed.returncode == 0 and completed.stderr == "", completed.stderr
    result = load_signed_json(output)
    assert result["signed_payload_sha256"] == SIGNATURES[version]
    assert [row["step"] for row in result["curve"]] == list(steps)
    expected_output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert output.read_text(encoding="utf-8") == expected_output
    summary = {"status": result["status"], "signature": result["signed_payload_sha256"]}
    assert completed.stdout == json.dumps(summary, sort_keys=True) + "\n"


@pytest.mark.parametrize("version", [2, 3, 4])
def test_versioned_verifier_help_keeps_description(version: int) -> None:
    completed = _run(SCRIPTS[version], "--help")
    assert completed.returncode == 0 and completed.stderr == ""
    assert DESCRIPTIONS[version] in completed.stdout
    assert "--bundle-root BUNDLE_ROOT" in completed.stdout
    assert "--output OUTPUT" in completed.stdout


def test_verification_failure_does_not_replace_output(tmp_path: Path) -> None:
    _build(tmp_path, 4)
    selection = tmp_path / "runs/stage-c4/warmstart-selection-v4/selection.json"
    _signed(
        selection,
        {
            "status": "failed",
            "selected_step": None,
            "evaluated_steps": list(STEPS[2]),
            "candidate_signed_payloads": {},
        },
    )
    output = tmp_path / "terminal-verification.json"
    output.write_text("preserve me\n", encoding="utf-8")
    completed = _run(SCRIPTS[4], "--bundle-root", str(tmp_path), "--output", str(output))
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "aggregate selection does not match candidate reports" in completed.stderr
    assert output.read_text(encoding="utf-8") == "preserve me\n"
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_scorer_pair_keeps_exact_hash_failure(tmp_path: Path) -> None:
    directory = tmp_path / "candidate"
    _scorer(tmp_path, directory, "action")
    sentinel_path = directory / "action-scores.verified.json"
    sentinel = load_signed_json(sentinel_path)
    unsigned = {key: value for key, value in sentinel.items() if key != "signed_payload_sha256"}
    unsigned["output_file_sha256"] = "0" * 64
    _signed(sentinel_path, unsigned)
    with pytest.raises(ValueError) as error:
        verify_scorer_pair(tmp_path, directory, "action")
    assert str(error.value) == f"{sentinel_path} has the wrong output file hash"
