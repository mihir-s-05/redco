"""Verify Phase-2 QA through disposable authenticated dependency trees."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_dependency_stack import (
    canonical_tree_manifest_bytes,
    materialize_live_owner_dependency_trees,
)
from redco.contracts import canonical_json

ROOT = Path(__file__).parents[1].resolve()
PARENT_COMMIT = "1ae6822b840858e2a27daeb44596342f2000979e"
EXPECTED_POST_TREES = {
    "renderers": "bd43d515c12dcaa1e1c0279941a1397d4ffba31a1557d6d7342a1322b195fcc4",
    "verifiers": "9dcf9e98dea73c2487d2165cd6cae35dc61fb66e00d377d85d5466886b3ea4e0",
}
EXPECTED_MODULE_BINDINGS = (
    {
        "module": "renderers.client",
        "relative_path": "renderers/client.py",
        "sha256": "f0ccece6b6da662c5823dc8a0334c876981b0733189b0ddbfa5b34094963d325",
        "bytes": 26590,
    },
    {
        "module": "verifiers.v1",
        "relative_path": "verifiers/v1/__init__.py",
        "sha256": "afc921822eff29263801f7af11e7d2b25cb1cad49c98ccf308103a923aaf7752",
        "bytes": 6653,
    },
    {
        "module": "verifiers.v1.clients.train",
        "relative_path": "verifiers/v1/clients/train.py",
        "sha256": "91be88e627c256cc7ee3823d74bb93a14d6c7bf3a81eed1705a831d06b92508f",
        "bytes": 22448,
    },
    {
        "module": "verifiers.v1.episode",
        "relative_path": "verifiers/v1/episode.py",
        "sha256": "6bef08c1879c7da0400f434f6cf4444165326f1f8aae02debf62470beb218e05",
        "bytes": 2301,
    },
    {
        "module": "verifiers.v1.trace",
        "relative_path": "verifiers/v1/trace.py",
        "sha256": "cdc7e1c310231c6b96fc864e9e696ba0061b6eb2ba87868ce5f55a398b73641c",
        "bytes": 26652,
    },
)
EXPECTED_OBSERVER_METHODS = (
    "abort",
    "after_raw_response",
    "after_response",
    "before_forward",
    "run_concurrent_children",
    "run_provider_call",
)
EXPECTED_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.12.3",
    "python_executable_sha256": (
        "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
    ),
    "uv_version": "uv 0.11.32 (x86_64-unknown-linux-gnu)",
    "uv_executable_sha256": (
        "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb"
    ),
    "uv_lock_sha256": (
        "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31"
    ),
    "offline": True,
    "project_sync": False,
}
SAMPLING_CONTRACT_SHA256 = (
    "819222244a81565a67331826be3dd362e14e1481043d60fccb569551a4471f6d"
)
TEST_NODES = (
    "tests/test_stage_d_source_env_pinned.py::"
    "test_bound_scientific_qa_missing_roster_fails_before_run_eval",
    "tests/test_stage_d_source_env_pinned.py::"
    "test_scientific_finalizer_timeout_is_bounded_and_side_effect_free",
    "tests/test_stage_d_source_env_pinned.py::"
    "test_real_bound_scientific_qa_runs_with_zero_provider_calls",
)
POSITIVE_EVIDENCE_FIELDS = {
    "schema_version",
    "domain",
    "receipt_sha256",
    "receipt_bytes",
    "report_sha256",
    "source_sha256",
    "trace_sha256",
    "recorded_action_digest",
    "commitment_receipt_sha256",
    "correspondence_receipt_sha256",
    "roster_sha256",
    "runtime_snapshot_sha256",
    "terminal_reply_sha256",
    "reward",
    "qa_receipt_record_count",
    "qa_barrier_record_count",
    "provider_dispatches",
    "generated_tokens",
    "judge_calls",
    "candidate_records",
    "scientific_execution_records",
    "qasper_rows_read",
    "launch_dataset_reads",
    "loopback_fixture_calls",
    "sampling_directions_drained",
    "campaign_recovery_callback_calls",
    "duplicate_run_eval_calls",
    "checkout_outputs_unchanged",
    "launch_claims_unchanged",
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


def _require_external_new_path(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(f"{name} must be outside the checkout")
    if path.exists() or path.is_symlink():
        raise ValueError(f"{name} already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"{name} parent must be a regular directory")
    return resolved


def _module_binding(module_name: str, component_root: Path) -> dict[str, object]:
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError(f"{module_name} does not resolve to a regular module file")
    path = Path(raw_path).resolve(strict=True)
    try:
        relative = path.relative_to(component_root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(
            f"{module_name} was imported outside its disposable dependency root"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{module_name} is not a regular dependency module")
    raw = path.read_bytes()
    return {
        "module": module_name,
        "relative_path": relative.as_posix(),
        "sha256": _sha256(raw),
        "bytes": len(raw),
    }


def _run_abi_probe(renderer_root: Path, verifier_root: Path) -> int:
    import renderers.client as renderer_client
    import verifiers.v1 as vf
    import verifiers.v1.clients.train as train_module
    import verifiers.v1.episode as episode_module
    import verifiers.v1.trace as trace_module
    from test_stage_d_source_producer import _episode
    from verifiers.v1.clients.train import split_engine_sampling
    from verifiers.v1.episode import WireEpisode
    from verifiers.v1.trace import ModelCall, Trace

    from redco.analysis.stage_d_replay_controller import (
        StageDReconstructionQAController,
        StageDReplayCallController,
    )
    from redco.analysis.stage_d_source_producer import (
        SAMPLING_CONTRACT,
    )
    from redco.analysis.stage_d_source_producer import (
        SAMPLING_CONTRACT_SHA256 as actual_sampling_contract_sha256,
    )
    from redco.integrations.verifiers_trace_v2 import extract_v2_rlm_provenance

    if sys.version_info[:3] != (3, 12, 3):
        raise RuntimeError("Phase-2 QA requires exact Python 3.12.3")
    if not callable(split_engine_sampling):
        raise RuntimeError("split_engine_sampling is absent from the disposable stack")
    if actual_sampling_contract_sha256 != SAMPLING_CONTRACT_SHA256:
        raise RuntimeError("the pinned 12-key sampling contract changed")
    fields = SAMPLING_CONTRACT.get("fields")
    if not isinstance(fields, list) or len(fields) != 12:
        raise RuntimeError("the pinned sampling contract is not the exact 12-key schema")

    episode_bytes = _episode(trace_id="phase2-qa-abi-probe")
    episode = WireEpisode.model_validate_json(episode_bytes)
    if not episode.traces or not isinstance(episode.traces[0], Trace):
        raise RuntimeError("the disposable Episode schema did not validate its Trace")
    trace = episode.traces[0]
    if not trace.calls or any(not isinstance(call, ModelCall) for call in trace.calls):
        raise RuntimeError("the disposable Trace schema did not validate ModelCall")
    raw_episode = json.loads(episode_bytes)
    raw_trace = cast(dict[str, Any], raw_episode["traces"][0])
    provenance = extract_v2_rlm_provenance(raw_trace)
    if len(provenance) != len(trace.calls) or any(
        record.trace_id != "phase2-qa-abi-probe" for record in provenance
    ):
        raise RuntimeError("the exact v2 provenance projection did not validate")
    if WireEpisode.model_validate(episode.model_dump(mode="json")) != episode:
        raise RuntimeError("the disposable Episode schema did not round-trip")

    observer = renderer_client.PreparedGenerateObserver
    required_observer_methods = set(EXPECTED_OBSERVER_METHODS)
    if any(not callable(getattr(observer, name, None)) for name in required_observer_methods):
        raise RuntimeError("the prepared-observer ABI is incomplete")
    for controller in (StageDReconstructionQAController, StageDReplayCallController):
        if any(
            not callable(getattr(controller, name, None))
            for name in ("run_provider_call", "run_concurrent_children")
        ):
            raise RuntimeError("a replay controller lacks its watchdog observer ABI")

    module_bindings = (
        _module_binding(renderer_client.__name__, renderer_root),
        _module_binding(vf.__name__, verifier_root),
        _module_binding(train_module.__name__, verifier_root),
        _module_binding(episode_module.__name__, verifier_root),
        _module_binding(trace_module.__name__, verifier_root),
    )
    if module_bindings != EXPECTED_MODULE_BINDINGS:
        raise RuntimeError("the disposable imported-module bindings changed")
    result = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d2-qa-disposable-abi-v1",
            "python": platform.python_version(),
            "split_engine_sampling": True,
            "sampling_contract_sha256": actual_sampling_contract_sha256,
            "sampling_field_count": 12,
            "episode_trace_model_call_v2_validated": True,
            "v2_provenance_record_count": len(provenance),
            "prepared_observer_methods": list(EXPECTED_OBSERVER_METHODS),
            "controller_watchdog_interface": True,
            "module_bindings": list(module_bindings),
        }
    )
    sys.stdout.buffer.write(result)
    return 0


def _pytest_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts != {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"Phase-2 QA pytest result changed: {counts}")
    return counts


def _validate_positive_evidence(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    sha_fields = (
        "receipt_sha256",
        "report_sha256",
        "source_sha256",
        "trace_sha256",
        "recorded_action_digest",
        "commitment_receipt_sha256",
        "correspondence_receipt_sha256",
        "roster_sha256",
        "runtime_snapshot_sha256",
        "terminal_reply_sha256",
    )
    if (
        not isinstance(value, dict)
        or set(value) != POSITIVE_EVIDENCE_FIELDS
        or canonical_json(value) != raw
        or value.get("schema_version") != 1
        or value.get("domain") != "redco-stage-d2-qa-positive-evidence-v1"
        or not isinstance(value.get("receipt_bytes"), int)
        or isinstance(value.get("receipt_bytes"), bool)
        or cast(int, value["receipt_bytes"]) <= 0
        or value.get("reward") != 1.0
        or value.get("loopback_fixture_calls") != 2
        or value.get("qa_receipt_record_count") != 1
        or value.get("qa_barrier_record_count") != 1
        or any(
            value.get(name) != 0
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
            value.get(name) is not True
            for name in (
                "sampling_directions_drained",
                "checkout_outputs_unchanged",
                "launch_claims_unchanged",
            )
        )
        or any(
            not isinstance(value.get(name), str)
            or len(cast(str, value[name])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in cast(str, value[name])
            )
            for name in sha_fields
        )
    ):
        raise RuntimeError("the genuine QA evidence is incomplete or noncanonical")
    return cast(dict[str, object], value)


def _verification_environment(
    first_root: Path,
    *,
    work_root: Path,
    evidence_path: Path,
) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}
    }
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(first_root / "renderers"),
                    str(first_root / "verifiers"),
                    str(ROOT / "src"),
                    str(ROOT / "tests"),
                    str(ROOT / "scripts"),
                    str(ROOT / "environments" / "redco_evidence_selection_v2"),
                )
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": str(work_root / "process-tmp"),
            "REDCO_PHASE2_QA_EVIDENCE_PATH": str(evidence_path),
        }
    )
    (work_root / "process-tmp").mkdir()
    return env


def _run_verification(args: argparse.Namespace) -> int:
    if _git_value("rev-parse", "HEAD") != PARENT_COMMIT:
        raise RuntimeError("Phase-2 QA verification must run at committed checkpoint 1ae6822")
    work_root = _require_external_new_path(Path(args.work_root), name="work root")
    work_root.mkdir()
    evidence_output = _require_external_new_path(
        Path(args.evidence_output), name="evidence output"
    )
    if evidence_output.parent != work_root:
        raise ValueError("evidence output must be a direct child of the work root")
    uv_executable = Path(args.uv_executable).resolve(strict=True)
    if uv_executable.is_symlink() or not uv_executable.is_file():
        raise ValueError("uv executable must be a regular file")
    first_root = work_root / "dependency-run-1"
    second_root = work_root / "dependency-run-2"
    first_payload = materialize_live_owner_dependency_trees(ROOT, first_root)
    second_payload = materialize_live_owner_dependency_trees(ROOT, second_root)
    if first_payload != second_payload:
        raise RuntimeError("the two dependency authentication runs differ")
    for component, expected in EXPECTED_POST_TREES.items():
        first_manifest = canonical_tree_manifest_bytes(first_root / component)
        second_manifest = canonical_tree_manifest_bytes(second_root / component)
        if first_manifest != second_manifest or _sha256(first_manifest) != expected:
            raise RuntimeError(f"disposable {component} trees are not byte-identical")

    positive_evidence_path = work_root / "positive-evidence.json"
    env = _verification_environment(
        first_root,
        work_root=work_root,
        evidence_path=positive_evidence_path,
    )
    abi = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--abi-probe",
            "--renderer-root",
            str(first_root / "renderers"),
            "--verifier-root",
            str(first_root / "verifiers"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    abi_value = json.loads(abi.stdout)
    if canonical_json(abi_value) != abi.stdout:
        raise RuntimeError("ABI probe output is not canonical")

    junit_path = work_root / "pytest-results.xml"
    pytest_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--capture=no",
            "--basetemp",
            str(work_root / "pytest"),
            f"--junitxml={junit_path}",
            *TEST_NODES,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if pytest_result.returncode != 0:
        sys.stderr.write(pytest_result.stdout)
        sys.stderr.write(pytest_result.stderr)
        raise RuntimeError("authenticated Phase-2 QA nodes failed")
    counts = _pytest_counts(junit_path)
    positive = _validate_positive_evidence(positive_evidence_path)
    uv_version = subprocess.run(
        [str(uv_executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_executable = Path(sys.executable).resolve(strict=True)
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_sha256": _sha256(python_executable.read_bytes()),
        "uv_version": uv_version,
        "uv_executable_sha256": _sha256(uv_executable.read_bytes()),
        "uv_lock_sha256": _sha256((ROOT / "uv.lock").read_bytes()),
        "offline": True,
        "project_sync": False,
    }
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError("the frozen Phase-2 CPU runtime binding changed")
    evidence = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d2-qa-disposable-overlay-evidence-v1",
            "parent_commit": PARENT_COMMIT,
            "parent_tree": _git_value("rev-parse", "HEAD^{tree}"),
            "dependency_authentication_runs": 2,
            "dependency_stack": first_payload,
            "abi_probe": abi_value,
            "runtime": runtime,
            "tests": {
                "nodes": list(TEST_NODES),
                "collected": counts["tests"],
                "passed": counts["tests"],
                "failed": counts["failures"] + counts["errors"],
                "skipped": counts["skipped"],
                "xfail": 0,
            },
            "positive": positive,
            "external_activity": {
                "prime_calls": 0,
                "provider_calls": 0,
                "model_calls": 0,
                "gpu_calls": 0,
                "wallet_calls": 0,
                "external_network_calls": 0,
                "loopback_fixture_calls": positive["loopback_fixture_calls"],
                "qasper_rows_read": 0,
                "parquet_access": False,
                "ordinal_181_accessed": False,
                "launch_dataset_reads": 0,
                "external_prime_rl_imported": False,
                "external_prime_rl_modified": False,
            },
        }
    )
    with evidence_output.open("xb") as output:
        output.write(evidence)
    print(
        json.dumps(
            {
                "evidence_sha256": _sha256(evidence),
                "positive_receipt_sha256": positive["receipt_sha256"],
                "tests_passed": counts["tests"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root")
    parser.add_argument("--evidence-output")
    parser.add_argument("--uv-executable")
    parser.add_argument("--abi-probe", action="store_true")
    parser.add_argument("--renderer-root")
    parser.add_argument("--verifier-root")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.abi_probe:
        if args.renderer_root is None or args.verifier_root is None:
            raise ValueError("ABI probe requires both disposable dependency roots")
        return _run_abi_probe(Path(args.renderer_root), Path(args.verifier_root))
    if any(
        value is None
        for value in (args.work_root, args.evidence_output, args.uv_executable)
    ):
        raise ValueError("verification requires work root, evidence output, and uv")
    return _run_verification(args)


if __name__ == "__main__":
    raise SystemExit(main())
