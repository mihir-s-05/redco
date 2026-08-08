"""Concrete one-attempt owner for the v13 support launch.

The launch module owns the immutable bundle.  This lower-level module owns the
mutable attempt, the external-process boundary, and the durable terminal
state.  It intentionally exchanges canonical intermediate artifacts with the
provider/branch runtime instead of accepting a preassembled result object.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_support_gate import (
    StageDSupportRules,
    verify_support_report,
)
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_draft_publication import (
    atomic_publish_set,
    validate_output_paths,
)
from redco.analysis.stage_d_v13_launch_lifecycle import (
    HANDOFF_V2_RESOURCE_KEYS,
    PROVISIONING_LEDGER_ENV,
    ProvisioningLedger,
    SigningIdentity,
    consume_execute_handoff_v2,
)
from redco.analysis.stage_d_v13_launch_observations import validate_prime_observation

OWNER_SEQUENCE = (
    "source_collection",
    "durable_target_roster",
    "k4_branch_replay",
    "deterministic_qasper_score",
    "support_gate",
    "canonical_support_report",
    "artifact_publication",
    "teardown",
    "terminal",
)
RUNTIME_ROOT_RELATIVE = "runs/stage-d/stage-d1-support-v13-launch/runtime"
EXECUTION_MANIFEST_NAME = "execution-manifest-v1.json"
TERMINATION_SECONDS = 2.0
_PREAUTHORIZED_ROOTS: set[Path] = set()


@dataclass(frozen=True, slots=True)
class SupportExecutionArtifacts:
    """Paths to output produced by the real external/branch owners."""

    manifest_path: Path


def _bundle() -> Any:
    # Importing lazily keeps the immutable bundle façade and this owner acyclic.
    from redco.analysis import stage_d_v13_support_launch as bundle

    return bundle


def _consume_bound_handoff(
    root: Path,
    observation_path: Path,
    capability_path: Path,
    hashes: Mapping[str, str],
) -> ProvisioningLedger:
    """Authenticate the signed handoff and create its provision claim once."""

    bundle = _bundle()
    observation_bytes = observation_path.read_bytes()
    validate_prime_observation(root, observation_path)
    observation = json.loads(observation_bytes)
    if not isinstance(observation, dict) or not isinstance(observation.get("resource"), dict):
        raise ValueError("validated Prime observation lacks a resource object")
    resource = cast(dict[str, Any], observation["resource"])
    if set(HANDOFF_V2_RESOURCE_KEYS).difference(resource):
        raise ValueError("validated Prime observation lacks handoff resource fields")
    resource_identity = {key: resource[key] for key in HANDOFF_V2_RESOURCE_KEYS}
    provisioning = ProvisioningLedger.open(root / bundle.LAUNCH_PROVISIONING_LEDGER_RELATIVE)
    signature_path = root / bundle.LAUNCH_HANDOFF_SIGNATURE_RELATIVE
    claim_path = root / bundle.LAUNCH_PROVISION_CLAIM_RELATIVE
    auth_value = json.loads((root / bundle.LAUNCH_AUTH_RELATIVE).read_bytes())
    if not isinstance(auth_value, dict):
        raise ValueError("launch authorization is not an object")
    identity = SigningIdentity.from_payload(auth_value.get("signing"))
    bundle_binding = {
        "commit": bundle._git_value(root, "rev-parse", "HEAD"),
        "tree": bundle._git_value(root, "rev-parse", "HEAD^{tree}"),
    }
    known_hosts_path = root / bundle.LAUNCH_KNOWN_HOSTS_RELATIVE
    if known_hosts_path.is_symlink() or not known_hosts_path.is_file():
        raise ValueError("campaign known_hosts is missing")
    known_hosts_sha256 = sha256_bytes(known_hosts_path.read_bytes())
    provisions = cast(list[dict[str, Any]], provisioning._state()["provisions"])
    if not provisions or not isinstance(provisions[-1].get("pod_id"), str):
        raise ValueError("provisioning ledger lacks the bound pod identity")
    consume_execute_handoff_v2(
        capability_path,
        signature_path,
        claim_path,
        identity=identity,
        bundle=bundle_binding,
        launch_authorization_sha256=hashes[bundle.LAUNCH_AUTH_RELATIVE],
        frozen_support_protocol_sha256=bundle.PROTOCOL_ROOT_SHA256,
        prime_observation_sha256=sha256_bytes(observation_bytes),
        resource_identity=resource_identity,
        pod_id=cast(str, provisions[-1]["pod_id"]),
        known_hosts_sha256=known_hosts_sha256,
        ledger=provisioning,
    )
    return provisioning


def authorize_handoff_before_runtime(
    root: Path,
    *,
    observation: Path,
    capability: Path,
    signature: Path,
) -> None:
    """Run the signed per-provision gate before model/runtime preparation."""

    bundle = _bundle()
    expected_capability = root.resolve() / bundle.LAUNCH_HANDOFF_RELATIVE
    expected_signature = root.resolve() / bundle.LAUNCH_HANDOFF_SIGNATURE_RELATIVE
    if capability.resolve() != expected_capability or signature.resolve() != expected_signature:
        raise ValueError("signed handoff paths are not the fixed production paths")
    hashes = bundle.verify_launch_bundle(root, require_post_commit=True)
    validate_output_paths(
        root,
        bundle._bundle_immutable_paths(root),
        output_paths=bundle.ATTEMPT_PATHS,
    )
    _consume_bound_handoff(root, observation, capability, hashes)
    _PREAUTHORIZED_ROOTS.add(root.resolve())


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_parent(path)
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def _replace_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.transition")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("attempt transition temporary path already exists")
    _exclusive_bytes(temporary, value)
    os.replace(temporary, path)
    _fsync_parent(path)


def _canonical_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or linked")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value), raw


def expected_branch_artifact_keys(roster: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive the branch-artifact key set from the durable target roster.

    The support cohort has 64 papers, but the number of branch artifacts is a
    property of the eligible target roster.  Ineligible papers are represented
    by explicit excluded-target dispositions and produce no branch artifact.
    """

    if set(roster) != {
        "schema_version",
        "domain",
        "planned_source_count",
        "completed_source_count",
        "eligible_source_count",
        "ineligible_source_count",
        "minimum_eligible_sources",
        "eligibility_passed",
        "source_sha256s",
        "targets",
        "excluded_targets",
    }:
        raise ValueError("target roster fields differ")
    targets = roster["targets"]
    excluded = roster["excluded_targets"]
    if not isinstance(targets, list) or not isinstance(excluded, list):
        raise ValueError("target roster targets are not lists")
    keys: list[str] = []
    source_target_counts: dict[str, int] = {}
    for item in targets:
        if not isinstance(item, dict) or not isinstance(item.get("group_id"), str):
            raise ValueError("target roster contains an invalid eligible target")
        if not isinstance(item.get("target_id"), str):
            raise ValueError("target roster contains an invalid eligible target")
        source_sha256 = item.get("source_sha256")
        if not isinstance(source_sha256, str) or not source_sha256:
            raise ValueError("eligible target lacks its source identity")
        keys.append(f"{item['group_id']}--{item['target_id']}.json")
        source_target_counts[source_sha256] = source_target_counts.get(source_sha256, 0) + 1
    for item in excluded:
        if not isinstance(item, dict) or not isinstance(item.get("ineligibility_reason"), str):
            raise ValueError("target roster lacks an ineligible disposition")
    if len(set(keys)) != len(keys):
        raise ValueError("target roster contains duplicate eligible targets")
    eligible_count = roster["eligible_source_count"]
    if type(eligible_count) is not int or eligible_count < 0:
        raise ValueError("target roster eligible count is invalid")
    if len(keys) != 2 * eligible_count:
        raise ValueError("target roster does not contain exactly two targets per eligible paper")
    if len(source_target_counts) != eligible_count or any(
        count != 2 for count in source_target_counts.values()
    ):
        raise ValueError("target roster target cardinality differs by eligible source")
    return tuple(sorted(keys))


def _safe_runtime_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"runtime artifact escapes the repository: {relative}") from error
    if path.is_symlink():
        raise ValueError(f"runtime artifact is a symlink: {relative}")
    return path


def _publish_execution_manifest(
    root: Path,
    *,
    runtime_root: Path,
    collection_path: Path,
    roster_path: Path,
    source_root: Path,
    branch_root: Path,
    ledger_root: Path,
    report_path: Path,
    provider_dispatch_observed: bool = False,
) -> Path:
    """Bind the durable outputs of the real collection/branch owners."""

    source_paths = tuple(sorted((source_root / "sources").glob("*.json")))
    if len(source_paths) != 64:
        raise RuntimeError("source owner did not publish the complete 64-row roster")
    source_artifacts: dict[str, dict[str, str]] = {}
    for path in source_paths:
        payload, raw = _canonical_object(path, label="source artifact")
        source_sha256 = payload.get("source_sha256")
        if not isinstance(source_sha256, str) or path.stem != source_sha256:
            raise RuntimeError("source artifact identity differs from its filename")
        source_artifacts[source_sha256] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_bytes(raw),
        }
    if len(source_artifacts) != 64:
        raise RuntimeError("source owner published duplicate source identities")

    roster, _roster_bytes = _canonical_object(roster_path, label="target roster")
    expected_branch_names = set(expected_branch_artifact_keys(roster))
    branch_paths = tuple(sorted(branch_root.glob("*.json")))
    if {path.name for path in branch_paths} != expected_branch_names:
        raise RuntimeError("branch artifact keys are not bijective with the target roster")
    branch_artifacts = {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_bytes(path.read_bytes()),
        }
        for path in branch_paths
    }

    record_paths = tuple(sorted((ledger_root / "records").glob("*.json")))
    if not record_paths:
        raise RuntimeError("ledger owner did not publish durable records")
    ledger_index = canonical_json_bytes(
        {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-ledger-index-v1",
            "records": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_bytes(path.read_bytes()),
                }
                for path in record_paths
            ],
        }
    )
    ledger_index_path = runtime_root / "ledger-index.json"
    _replace_bytes(ledger_index_path, ledger_index)

    manifest = canonical_json_bytes(
        {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-execution-manifest-v1",
            "owner_sequence": list(OWNER_SEQUENCE),
            "collection_receipt": {
                "path": collection_path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(collection_path.read_bytes()),
            },
            "target_roster": {
                "path": roster_path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(roster_path.read_bytes()),
            },
            "source_artifacts": source_artifacts,
            "branch_artifacts": branch_artifacts,
            "support_report": {
                "path": report_path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(report_path.read_bytes()),
            },
            "ledger": {
                "path": ledger_index_path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(ledger_index),
            },
            "provider_dispatch_observed": provider_dispatch_observed,
            "source_count": 64,
            "branch_count_k": 4,
            "branch_target_count": len(expected_branch_names),
            "branch_target_artifact_bijection": True,
            "retry": False,
        }
    )
    manifest_path = runtime_root / EXECUTION_MANIFEST_NAME
    _replace_bytes(manifest_path, manifest)
    return manifest_path


def _verify_execution_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    bundle = _bundle()
    manifest_path = manifest_path.resolve(strict=False)
    try:
        manifest_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("execution manifest escapes the repository") from error
    manifest, manifest_bytes = _canonical_object(manifest_path, label="execution manifest")
    expected_keys = {
        "schema_version",
        "domain",
        "owner_sequence",
        "collection_receipt",
        "target_roster",
        "source_artifacts",
        "branch_artifacts",
        "support_report",
        "ledger",
        "provider_dispatch_observed",
        "source_count",
        "branch_count_k",
        "branch_target_count",
        "branch_target_artifact_bijection",
        "retry",
    }
    if set(manifest) != expected_keys:
        raise ValueError("execution manifest fields differ")
    if (
        manifest["schema_version"] != 1
        or manifest["domain"] != "redco-stage-d1-support-v13-execution-manifest-v1"
        or tuple(manifest["owner_sequence"]) != OWNER_SEQUENCE
        or manifest["provider_dispatch_observed"] is not True
        or manifest["source_count"] != 64
        or manifest["branch_count_k"] != 4
        or manifest["branch_target_artifact_bijection"] is not True
        or manifest["retry"] is not False
    ):
        raise ValueError("execution manifest does not prove the fixed production owner sequence")

    def bound_file(item: object, label: str) -> tuple[Path, str]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"{label} binding is incomplete")
        relative = item["path"]
        expected = item["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"{label} binding types differ")
        path = _safe_runtime_path(root, relative)
        value = path.read_bytes()
        if sha256_bytes(value) != expected:
            raise ValueError(f"{label} bytes differ")
        return path, expected

    collection_path, collection_sha = bound_file(
        manifest["collection_receipt"], "collection receipt"
    )
    roster_path, roster_sha = bound_file(manifest["target_roster"], "target roster")
    roster, _roster_bytes = _canonical_object(roster_path, label="target roster")
    expected_branch_names = set(expected_branch_artifact_keys(roster))
    report_path, report_sha = bound_file(manifest["support_report"], "support report")
    ledger_path, ledger_sha = bound_file(manifest["ledger"], "ledger")
    _canonical_object(collection_path, label="collection receipt")
    _canonical_object(roster_path, label="target roster")
    ledger_index, _ledger_index_bytes = _canonical_object(ledger_path, label="ledger index")
    if set(ledger_index) != {"schema_version", "domain", "records"}:
        raise ValueError("ledger index fields differ")
    records = ledger_index["records"]
    if (
        ledger_index["schema_version"] != 1
        or ledger_index["domain"] != "redco-stage-d1-support-v13-ledger-index-v1"
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("ledger index is incomplete")
    seen_records: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("ledger index record is malformed")
        record_path = record["path"]
        record_sha = record["sha256"]
        if not isinstance(record_path, str) or not isinstance(record_sha, str):
            raise ValueError("ledger index record types differ")
        if record_path in seen_records:
            raise ValueError("ledger index contains duplicate records")
        seen_records.add(record_path)
        bound_file({"path": record_path, "sha256": record_sha}, "ledger record")
    source_artifacts = manifest["source_artifacts"]
    branch_artifacts = manifest["branch_artifacts"]
    if not isinstance(source_artifacts, dict) or not isinstance(branch_artifacts, dict):
        raise ValueError("execution artifact rosters are not objects")
    for roster in (source_artifacts, branch_artifacts):
        for name, item in roster.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                raise ValueError("execution artifact roster entry is malformed")
            bound_file(item, "execution artifact")
    report, report_bytes = _canonical_object(report_path, label="support report")
    rules = StageDSupportRules.from_bytes(
        bundle._read_bound(
            root,
            bundle.FROZEN_SUPPORT_RULES_RELATIVE,
            bundle.FROZEN_SUPPORT_RULES_SHA256,
        )
    )
    source_hashes = tuple(sorted(str(item) for item in manifest["source_artifacts"]))
    artifact_hashes = tuple(
        sorted(str(item["sha256"]) for item in manifest["branch_artifacts"].values())
    )
    verified_report_sha, decision = verify_support_report(
        report_bytes,
        expected_rules_sha256=rules.rules_sha256,
        source_sha256s=source_hashes,
        artifact_sha256s=artifact_hashes,
    )
    if verified_report_sha != report_sha or report.get("required_papers") != 64:
        raise ValueError("support report is not the authenticated 64-paper gate report")
    nested = report.get("nested_support")
    if not isinstance(nested, dict) or set(nested) < {"N_scaffold", "N_eligible", "N_joint"}:
        raise ValueError("support report lacks canonical support counts")
    if {
        Path(str(name)).name for name in branch_artifacts
    } != expected_branch_names:
        raise ValueError("branch artifact roster is not a target bijection")
    if manifest["branch_target_count"] != len(expected_branch_names):
        raise ValueError("branch artifact count differs from the authenticated roster")
    if len(source_artifacts) != 64:
        raise ValueError("execution manifest does not bind complete support artifacts")
    return {
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "collection_receipt_sha256": collection_sha,
        "target_roster_sha256": roster_sha,
        "support_report_sha256": report_sha,
        "ledger_sha256": ledger_sha,
        "decision": decision,
        "N_scaffold": nested["N_scaffold"],
        "N_eligible": nested["N_eligible"],
        "N_joint": nested["N_joint"],
        "provider_dispatch_observed": True,
        "manifest_path": str(manifest_path.relative_to(root.resolve())),
        "collection_receipt_path": str(collection_path.relative_to(root.resolve())),
        "target_roster_path": str(roster_path.relative_to(root.resolve())),
        "support_report_path": str(report_path.relative_to(root.resolve())),
        "ledger_path": str(ledger_path.relative_to(root.resolve())),
    }


class ProductionSupportActuator:
    """Fixed process boundary for source, branch, scorer, and gate owners.

    The actual deployment supplies the already-authenticated runtime process.
    This class has no scientific knobs: it consumes the fixed launch plan and
    waits for the canonical execution-manifest handoff produced by the existing
    source/branch/scorer owners.  Tests may replace only ``_run_owned_process``
    to write that handoff; they cannot inject a completed result object.
    """

    def __init__(
        self,
        root: Path,
        *,
        dispatch_marker: Callable[[str], None] | None = None,
        provisioning_path: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self._processes: list[subprocess.Popen[bytes]] = []
        self.provider_dispatch_observed = False
        self._dispatch_marker = dispatch_marker
        self._provisioning_path = (
            None if provisioning_path is None else provisioning_path.resolve()
        )

    def _refresh_dispatch_state(self) -> None:
        if self._provisioning_path is not None:
            self.provider_dispatch_observed = ProvisioningLedger.open(
                self._provisioning_path
            ).has_provider_post()

    def mark_provider_dispatch(self, operation_id: str) -> None:
        """Mark the irreversible boundary immediately before a provider POST."""

        if self.provider_dispatch_observed:
            raise RuntimeError("provider dispatch has already been observed")
        if self._dispatch_marker is None:
            raise RuntimeError("provider POST owner is not connected to the attempt ledger")
        self._dispatch_marker(operation_id)
        self.provider_dispatch_observed = True

    def _run_owned_process(self, plan_bytes: bytes, source_eval: bytes) -> None:
        bundle = _bundle()
        runtime_root = self.root / RUNTIME_ROOT_RELATIVE
        runtime_root.mkdir(parents=True, exist_ok=True)
        protocol_path = self.root / bundle.LAUNCH_PROTOCOL_RELATIVE
        config_path = self.root / bundle.LAUNCH_SOURCE_EVAL_RELATIVE
        branch_config_path = self.root / bundle.LAUNCH_BRANCH_RUNTIME_RELATIVE
        collection_script = self.root / "scripts/run_stage_d_source_collection.py"
        scientific_script = self.root / "scripts/run_stage_d_scientific_campaign.py"
        if not collection_script.is_file():
            raise RuntimeError("authenticated source-collection owner is missing")
        if not scientific_script.is_file():
            raise RuntimeError("authenticated branch/replay owner is missing")
        if not branch_config_path.is_file() or branch_config_path.is_symlink():
            raise RuntimeError("authenticated one-by-one branch runtime is missing")
        plan = bundle.StageDCollectionPlan.from_bytes(plan_bytes)
        try:
            source_config = tomllib.loads(source_eval.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError("authenticated support source config is invalid") from error
        env = source_config.get("env")
        if not isinstance(env, dict):
            raise RuntimeError("authenticated support source config lacks its environment")
        required_env = {
            "config_sha256",
            "ledger_path",
            "artifact_path",
            "master_seed",
            "preregistration_sha256",
            "source_sha256",
            "runtime_sha256",
        }
        if set(required_env) - set(env):
            raise RuntimeError("authenticated support source config is incomplete")
        ledger_relative = env["ledger_path"]
        source_relative = env["artifact_path"]
        master_seed = env["master_seed"]
        if not all(
            isinstance(value, str)
            for value in (ledger_relative, source_relative, master_seed)
        ):
            raise RuntimeError("authenticated support source config has invalid paths")
        target_root = runtime_root / "target-roster"
        branch_root = runtime_root / "branch-results"
        episode_root = runtime_root / "episodes"
        command = [
            sys.executable,
            str(collection_script),
            "--config",
            str(config_path),
            "--config-sha256",
            sha256_bytes(source_eval),
            "--protocol-manifest",
            str(protocol_path),
            "--protocol-manifest-sha256",
            sha256_bytes(protocol_path.read_bytes()),
            "--genesis-config-sha256",
            str(env["config_sha256"]),
            "--preregistration-sha256",
            str(env["preregistration_sha256"]),
            "--source-sha256",
            str(env["source_sha256"]),
            "--runtime-sha256",
            str(env["runtime_sha256"]),
            "--plan-sha256",
            plan.plan_sha256,
            "--plan-output",
            str(runtime_root / "collection-plan.json"),
            "--receipt-output",
            str(runtime_root / "collection-receipt.json"),
            "--dependency-stack",
            str(self.root / bundle.DEPENDENCY_MANIFEST_RELATIVE),
            "--rlm-archive",
            bundle.OFFLINE_RLM_BINDINGS["checkout_archive_path"],
            "--uv-binary",
            bundle.OFFLINE_RLM_BINDINGS["checkout_uv_path"],
            "--uv-cache-archive",
            bundle.OFFLINE_RLM_BINDINGS["checkout_cache_archive_path"],
            "--rlm-launcher",
            bundle.OFFLINE_RLM_BINDINGS["checkout_launcher_path"],
            "--branch-artifacts",
            str(target_root),
            "--support-rules",
            str(self.root / bundle.FROZEN_SUPPORT_RULES_RELATIVE),
            "--support-rules-sha256",
            bundle.FROZEN_SUPPORT_RULES_SHA256,
        ]
        campaign_deadline = time.monotonic() + 21_600

        def run_owned(command: list[str], owner: str) -> None:
            process_environment = os.environ.copy()
            source_root = self.root / "src"
            if not source_root.is_dir():
                source_root = Path(__file__).resolve().parents[2]
            inherited_pythonpath = process_environment.get("PYTHONPATH")
            process_environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(source_root), inherited_pythonpath)
                if part
            )
            if self._provisioning_path is not None:
                process_environment[PROVISIONING_LEDGER_ENV] = str(
                    self._provisioning_path
                )
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=process_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._processes.append(process)
            try:
                remaining = campaign_deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 0)
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                process.terminate()
                try:
                    process.wait(timeout=TERMINATION_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                self._refresh_dispatch_state()
                raise TimeoutError(f"{owner} exceeded the frozen campaign bound") from error
            if process.returncode != 0:
                self._refresh_dispatch_state()
                message = stderr.decode("utf-8", "replace")[-2000:]
                raise RuntimeError(f"{owner} failed: {message}")
            self._refresh_dispatch_state()
            del stdout

        run_owned(command, "source collection owner")
        branch_root.mkdir(parents=True, exist_ok=True)
        episode_root.mkdir(parents=True, exist_ok=True)
        branch_runtime_bytes = branch_config_path.read_bytes()
        scientific_command = [
            sys.executable,
            str(scientific_script),
            "--config",
            str(branch_config_path),
            "--config-sha256",
            sha256_bytes(branch_runtime_bytes),
            "--protocol-manifest",
            str(protocol_path),
            "--protocol-manifest-sha256",
            sha256_bytes(protocol_path.read_bytes()),
            "--ledger",
            str(self.root / ledger_relative),
            "--source-artifacts",
            str(self.root / source_relative),
            "--artifact-output",
            str(branch_root),
            "--episode-output",
            str(episode_root),
            "--master-seed",
            master_seed,
            "--dependency-stack",
            str(self.root / bundle.DEPENDENCY_MANIFEST_RELATIVE),
            "--rlm-archive",
            bundle.OFFLINE_RLM_BINDINGS["checkout_archive_path"],
            "--uv-binary",
            bundle.OFFLINE_RLM_BINDINGS["checkout_uv_path"],
            "--uv-cache-archive",
            bundle.OFFLINE_RLM_BINDINGS["checkout_cache_archive_path"],
            "--rlm-launcher",
            bundle.OFFLINE_RLM_BINDINGS["checkout_launcher_path"],
            "--support-report",
            str(self.root / bundle.LAUNCH_SUPPORT_REPORT_RELATIVE),
            "--support-rules",
            str(self.root / bundle.FROZEN_SUPPORT_RULES_RELATIVE),
            "--support-rules-sha256",
            bundle.FROZEN_SUPPORT_RULES_SHA256,
        ]
        run_owned(scientific_command, "branch/replay and support-gate owner")
        manifest_path = _publish_execution_manifest(
            self.root,
            runtime_root=runtime_root,
            collection_path=runtime_root / "collection-receipt.json",
            roster_path=target_root / "target-roster.json",
            source_root=self.root / source_relative,
            branch_root=branch_root,
            ledger_root=self.root / ledger_relative,
            report_path=self.root / bundle.LAUNCH_SUPPORT_REPORT_RELATIVE,
            provider_dispatch_observed=self.provider_dispatch_observed,
        )
        if not manifest_path.is_file():
            raise RuntimeError("support owners did not publish the execution manifest")

        return

    def run_once(self, plan_bytes: bytes, source_eval: bytes) -> SupportExecutionArtifacts:
        self._run_owned_process(plan_bytes, source_eval)
        self._refresh_dispatch_state()
        bundle = _bundle()
        manifest_path = self.root / bundle.LAUNCH_RUNTIME_MANIFEST_RELATIVE
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RuntimeError("production owners did not publish the execution manifest")
        return SupportExecutionArtifacts(manifest_path)

    def teardown(self) -> None:
        for process in tuple(self._processes):
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=TERMINATION_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._processes.clear()
        self._refresh_dispatch_state()


def _terminal_payload(
    *,
    state: str,
    provider_dispatch_observed: bool,
    error: BaseException | None = None,
    evidence: dict[str, Any] | None = None,
) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-launch-terminal-v1",
        "state": state,
        "attempt": 1,
        "retry": False,
        "provider_dispatch_observed": provider_dispatch_observed,
        "science_authorized": False,
        "training_authorized": False,
        "heldout_evaluation_authorized": False,
        "scientific_transition_authorized": False,
        "redeployment_allowed": not provider_dispatch_observed,
    }
    if error is not None:
        payload["error_type"] = type(error).__qualname__
        payload["error_message_sha256"] = sha256_bytes(str(error).encode("utf-8"))
    if evidence is not None:
        payload["evidence"] = evidence
    return cast(bytes, canonical_json_bytes(payload))


terminal_payload = _terminal_payload


def execute_support_once(
    root: Path,
    *,
    preflight_observation: Path | None = None,
    pod_runtime_observation: Path | None = None,
    preflight_snapshot: Path | None = None,
    capability: Path | None = None,
    capability_signature: Path | None = None,
) -> dict[str, str]:
    """Run exactly one post-commit support attempt through real artifact owners."""

    bundle = _bundle()
    root = root.resolve()
    observation_path = preflight_observation or preflight_snapshot
    if observation_path is None:
        raise ValueError("execute-once requires an authenticated Prime observation")
    synthetic = preflight_snapshot is not None and preflight_observation is None
    if not synthetic:
        if observation_path.resolve() != root / bundle.LAUNCH_PRIME_OBSERVATION_RELATIVE:
            raise ValueError("production execution requires the fixed Prime handoff path")
        if pod_runtime_observation is None or pod_runtime_observation.resolve() != (
            root / bundle.LAUNCH_POD_OBSERVATION_RELATIVE
        ):
            raise ValueError("production execution requires the fixed pod observation path")
        expected_capability = root / bundle.LAUNCH_HANDOFF_RELATIVE
        if capability is None or capability.resolve() != expected_capability:
            raise ValueError("production execution requires the fixed execute handoff path")
        expected_signature = root / bundle.LAUNCH_HANDOFF_SIGNATURE_RELATIVE
        if (
            capability_signature is None
            or capability_signature.resolve() != expected_signature
        ):
            raise ValueError("production execution requires the fixed handoff signature path")
    output_paths = (*bundle.ATTEMPT_PATHS, bundle.LAUNCH_RUNTIME_MANIFEST_RELATIVE)
    capability_path = capability
    hashes: dict[str, str]
    attempt_path = root / bundle.LAUNCH_ATTEMPT_RELATIVE
    if attempt_path.exists() or attempt_path.is_symlink():
        raise RuntimeError("support attempt has already been consumed")
    provisioning: ProvisioningLedger | None = None
    if synthetic:
        raise ValueError("synthetic preflight cannot authorize execute-once")
    if capability_path is None or capability_signature is None:
        raise ValueError("signed execute handoff is missing")
    hashes = bundle.verify_launch_bundle(root, require_post_commit=True)
    validate_output_paths(
        root,
        bundle._bundle_immutable_paths(root),
        output_paths=output_paths,
    )
    preauthorized = root in _PREAUTHORIZED_ROOTS
    if preauthorized:
        _PREAUTHORIZED_ROOTS.remove(root)
        claim_path = root / bundle.LAUNCH_PROVISION_CLAIM_RELATIVE
        if claim_path.is_symlink() or not claim_path.is_file():
            raise ValueError("pre-runtime handoff claim is missing")
        provisioning = ProvisioningLedger.open(
            root / bundle.LAUNCH_PROVISIONING_LEDGER_RELATIVE
        )
    else:
        provisioning = _consume_bound_handoff(
            root,
            observation_path,
            capability_path,
            hashes,
        )

    def dispatch_marker(operation_id: str) -> None:
        assert provisioning is not None
        provisioning.record_provider_post(operation_id=operation_id)
    hashes = bundle.preflight_validate(
        root,
        observation_path,
        require_post_commit=True,
        runtime_observation_path=pod_runtime_observation,
        synthetic=False,
    )
    plan_bytes = (root / bundle.LAUNCH_PLAN_RELATIVE).read_bytes()
    source_eval = (root / bundle.LAUNCH_SOURCE_EVAL_RELATIVE).read_bytes()
    claim = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-launch-attempt-v1",
        "state": "verified_preflight_claimed",
        "attempt": 1,
        "attempt_limit": 1,
        "retry": False,
        "authorization_sha256": hashes[bundle.LAUNCH_AUTH_RELATIVE],
        "collection_plan_sha256": sha256_bytes(plan_bytes),
        "provider_dispatch_observed": False,
        "evidence_observed": False,
        "closed": False,
    }
    owner = ProductionSupportActuator(
        root,
        dispatch_marker=dispatch_marker,
        provisioning_path=(None if provisioning is None else provisioning.path),
    )
    dispatch_observed = False
    teardown_attempted = False
    try:
        _exclusive_bytes(attempt_path, canonical_json_bytes(claim))
        claim["state"] = "provider_dispatch_boundary"
        claim["provider_dispatch_observed"] = False
        _replace_bytes(attempt_path, canonical_json_bytes(claim))
        artifacts = owner.run_once(plan_bytes, source_eval)
        dispatch_observed = bool(getattr(owner, "provider_dispatch_observed", True))
        claim["provider_dispatch_observed"] = dispatch_observed
        claim["state"] = "evidence_observed"
        claim["evidence_observed"] = True
        _replace_bytes(attempt_path, canonical_json_bytes(claim))
        evidence = _verify_execution_manifest(root, artifacts.manifest_path)
        if provisioning is not None:
            provisioning.record_evidence()
        claim["state"] = "closed_before_terminal_publication"
        claim["closed"] = True
        _replace_bytes(attempt_path, canonical_json_bytes(claim))
        owner.teardown()
        teardown_attempted = True
        provisioning.close()
        terminal = _terminal_payload(
            state="completed_support_only",
            provider_dispatch_observed=dispatch_observed,
            evidence=evidence,
        )
        report_path = root / evidence["support_report_path"]
        report = report_path.read_bytes()
        payloads = {
            bundle.LAUNCH_TERMINAL_RELATIVE: terminal,
            bundle.LAUNCH_SUPPORT_REPORT_RELATIVE: report,
        }
        return cast(
            dict[str, str],
            atomic_publish_set(
                root,
                payloads,
                immutable_paths=bundle._bundle_immutable_paths(root),
                manifest_path=bundle.LAUNCH_TERMINAL_RELATIVE,
                require_draft_envelope=False,
            ),
        )
    except BaseException as error:
        dispatch_observed = bool(getattr(owner, "provider_dispatch_observed", True))
        claim["state"] = "closed_before_terminal_publication"
        claim["closed"] = True
        claim["provider_dispatch_observed"] = dispatch_observed
        with contextlib.suppress(BaseException):
            _replace_bytes(attempt_path, canonical_json_bytes(claim))
        teardown_error: BaseException | None = None
        if not teardown_attempted:
            teardown_attempted = True
            try:
                owner.teardown()
            except BaseException as caught:
                teardown_error = caught
        with contextlib.suppress(BaseException):
            provisioning.close()
        if teardown_error is not None:
            raise error.with_traceback(error.__traceback__) from teardown_error
        terminal_path = root / bundle.LAUNCH_TERMINAL_RELATIVE
        if terminal_path.exists() or terminal_path.is_symlink():
            raise RuntimeError("support terminal disposition already exists") from error
        terminal = _terminal_payload(
            state=(
                "failed_terminal_no_retry"
                if dispatch_observed
                else "failed_zero_call_pre_dispatch"
            ),
            provider_dispatch_observed=dispatch_observed,
            error=error,
        )
        try:
            atomic_publish_set(
                root,
                {bundle.LAUNCH_TERMINAL_RELATIVE: terminal},
                immutable_paths=bundle._bundle_immutable_paths(root),
                manifest_path=bundle.LAUNCH_TERMINAL_RELATIVE,
                require_draft_envelope=False,
            )
        except BaseException as publication_error:
            raise error.with_traceback(error.__traceback__) from publication_error
        if teardown_error is not None:
            raise error.with_traceback(error.__traceback__) from teardown_error
        raise


__all__ = [
    "OWNER_SEQUENCE",
    "ProductionSupportActuator",
    "SupportExecutionArtifacts",
    "execute_support_once",
    "expected_branch_artifact_keys",
    "terminal_payload",
]
