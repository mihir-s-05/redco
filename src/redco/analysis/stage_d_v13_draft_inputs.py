"""Read-only authentication and dependency bindings for the v13 draft.

This module deliberately has no publication side effects.  It authenticates
the pre-outcome inputs, reconstructs the pinned Prime component in a temporary
clean tree, and returns bounded archive facts for the draft builder.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_dependency_stack import canonical_tree_manifest_bytes
from redco.analysis.stage_d_v13_draft import sha256_bytes, sha256_json

REPAIR_COMMIT = "8b64ad0a9f443801a2cb5a00bf18da5335f9a82f"
REPAIRED_SOURCE_SHA256 = "2e13f156b9dd078ce02bb06eeeb9a69f122b8fe25c48c05abce3290d702ee522"
PRE_REPAIR_SOURCE_SHA256 = "4e2f59b3ae973eaaa8aab8dca378e196430acb650cf8c005dbb2227b1d0923b1"
V12_ARCHIVE_SHA256 = "c2bb6713234653fc08e10778aa17815ad5a26f769406806037f50c390820b894"
V12_EVIDENCE_MANIFEST_SHA256 = "90c694a45ece9887ea658adc36a8b30f6b7d78f8b111c017f54b5e5d51003671"
V12_TERMINAL_REPORT_SHA256 = "3b79bbf541ee4210744b37f2df49531ca4e6b601f500e0ecaa43e3fa9e8ca9ec"
V12_AUDIT_REPORT_SHA256 = "97f743f9dfee0c5f2988073dc00efc7a83765698ead298d89c9f9ae26714a588"
V12_PROTOCOL_SHA256 = "2be6b64916ef3620dc15fade89916b616de1ea8f54db0109c7f0ff5c3be8e9fd"
V12_PREREG_SHA256 = "8cf086e4b198c45306a0cb7d3289b72e65fa2ee9ae34940a42e141542003b429"
V12_PLAN_SHA256 = "9870c18fd43ee3a08bb212d5f9b506104e39b3d2dc2ccd7b689240799d2696cb"
V12_SOURCE_SHA256 = "034a9bc05d8ff28699d29e6e6d649dbfb9cb57191af0fb6af34983f9d18d9141"
V12_SOURCE_EVAL_SHA256 = "704eca5bb15a7ee52572653110639857dcffe422a2d5cfe1a66b498959e88351"
V12_DEPENDENCY_STACK_SHA256 = "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831"
V12_PYPROJECT_SHA256 = "94c85ca6ffd627b07cfee14ce8ba80b3cb19fb279e7b98792fddf12695e0699b"
V12_UV_LOCK_SHA256 = "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31"
V12_SELECTED_MANIFEST_SHA256 = "c02dbbdd03b5322d91ab350bcbdf4abe2ff10f4ecf61ba17ce78bacdcb0f95ba"
V12_GREEN_MANIFEST_SHA256 = "e1961cb190bbfa5325e29ef25a228440194c3fbce9d08a589838ebea172ad681"
POST_REPAIR_AUDIT_SHA256 = "f17305ed9242c60921f5265330ebb154a94555980de48d83696f79031e671e34"
POST_REPAIR_REGRESSION_SHA256 = "f8c45493c60c2b84179b13e5761c542b204791c078ec6c0afb3cf932f5a568c5"
V6_SUCCESSOR_DATASET_SHA256 = "153c25a1697737d4df58883adedf55e056d6cd58f08f86e2489391b40b5183ac"
V6_SUCCESSOR_MANIFEST_SHA256 = "5b1667fa9f17c7e733276b17534de6453598e60b8e52a733a2028c7dab671697"
V6_ADDRESS_AUDIT_SHA256 = "8f0e081fe2b3ed8b15254ac5068569d9e0a493fc1e8b832d7956af43ed1f50dd"
V5_SUCCESSOR_DATASET_SHA256 = "bb576082ba15535d7b0a996ea5c14dd008ebde634a0d8c5c7258f81d5ac9577d"
V4_DATASET_SHA256 = "88fa2c114d2f251b8ce0400023980fe652e4733d14b0357f5517f517d5775d71"
HISTORICAL_ADDRESS_HASHES: dict[str, str] = {
    "reports/stage-d1-support-successor-address-audit-v1.json": (
        "ee5fc07b6bf76d470d9bcf26e1085d3055a283fb3df3212359353d5b2586d6df"
    ),
    "reports/stage-d1-support-successor-address-audit-v2.json": (
        "c44eb9cdd16575b2f91fa75e61a93aa421a1dacc7f190bbcde85c558353c6646"
    ),
    "reports/stage-d1-support-successor-address-audit-v3.json": (
        "c3902886c4dca2ba85f58b2cc81aa1267da4eb67646d9aaf959322082a332c67"
    ),
    "reports/stage-d1-support-successor-address-audit-v4.json": (
        "a03784d33cf8b9fbedf5b80685445d477f327fc0bcc99b20be9f3d85711d7505"
    ),
    "reports/stage-d1-support-successor-address-audit-v5.json": (
        "57969284590567c19e0100e1bb23e1639fb490c2e040be68f262aedab4182db3"
    ),
    "reports/stage-d1-support-successor-address-audit-v6.json": V6_ADDRESS_AUDIT_SHA256,
}
HISTORICAL_ROLLOUT_HASHES: dict[str, str] = {
    "reports/stage-d1-support-successor-preregistration-audit-v8.json": (
        "ec6c21ad349fbef55293c6225d5d520a9c78f3b01d084be0ffe962c8fe2f0c46"
    ),
    "reports/stage-d1-support-successor-preregistration-audit-v9.json": (
        "5236b60a9bc03cbb0d0d19a7025f023a59bb8c14630460cad7bc1d9376685828"
    ),
    "reports/stage-d1-support-successor-preregistration-audit-v10.json": (
        "d22bd43b267f12569f6bc2524c6e254bde44510692cafb045f98f98ac29e7aac"
    ),
    "reports/stage-d1-support-v8-terminal.json": (
        "b8d0199debe65758c3e9b831cdc462a77fc76188ebac04d9e974dda6f2924c73"
    ),
    "reports/stage-d1-support-v9-terminal.json": (
        "f7203c119d89bdcc11178fcc0233f8e3f0e9e14d06600b934e25bbfb3c7cbb7b"
    ),
    "reports/stage-d1-support-v10-terminal.json": (
        "e49478bc76ad82e8beccb1e392edcb4352f58169fec67a0f072cd323647177d9"
    ),
}

POST_REPAIR_HASHES = {
    "reports/stage-d1-v12-post-repair-audit-v1.json": POST_REPAIR_AUDIT_SHA256,
    "reports/stage-d1-source-comparison-post-repair-v1.json": POST_REPAIR_REGRESSION_SHA256,
}

FROZEN_HASHES: dict[str, str] = {
    "runs/stage-d/stage-d1-support-v12-terminal.tar.gz": V12_ARCHIVE_SHA256,
    "runs/stage-d/stage-d1-support-v12-evidence-sha256.txt": V12_EVIDENCE_MANIFEST_SHA256,
    "reports/stage-d1-support-v12-terminal.json": V12_TERMINAL_REPORT_SHA256,
    "reports/stage-d1-support-v12-finalization-audit-v1.json": V12_AUDIT_REPORT_SHA256,
    "reports/stage-d1-source-comparison-selected-tests-v1.json": V12_SELECTED_MANIFEST_SHA256,
    "reports/stage-d1-source-comparison-green-suite-v1.json": V12_GREEN_MANIFEST_SHA256,
    "reports/stage-d1-support-successor-address-audit-v6.json": V6_ADDRESS_AUDIT_SHA256,
    "configs/stage-d/stage-d1-support-protocol-v12.json": V12_PROTOCOL_SHA256,
    "configs/stage-d/stage-d1-support-preregistration-v12.json": V12_PREREG_SHA256,
    "configs/stage-d/stage-d1-support-collection-plan-v11.json": V12_PLAN_SHA256,
    "configs/stage-d/stage-d1-support-source-v12.json": V12_SOURCE_SHA256,
    "configs/stage-d/stage-d1-support-source-eval-v12.toml": V12_SOURCE_EVAL_SHA256,
    "configs/stage-d/stage-d1-dependency-stack-v12.json": V12_DEPENDENCY_STACK_SHA256,
    "datasets/stage-d/qasper-support-successor-v6.jsonl": V6_SUCCESSOR_DATASET_SHA256,
    "datasets/stage-d/qasper-support-successor-manifest-v6.json": V6_SUCCESSOR_MANIFEST_SHA256,
    "datasets/stage-d/qasper-support-successor-v5.jsonl": V5_SUCCESSOR_DATASET_SHA256,
    "datasets/stage-d/qasper-deterministic-v4.jsonl": V4_DATASET_SHA256,
    "pyproject.toml": V12_PYPROJECT_SHA256,
    "uv.lock": V12_UV_LOCK_SHA256,
    **HISTORICAL_ADDRESS_HASHES,
    **HISTORICAL_ROLLOUT_HASHES,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return value


_HISTORICAL_IDENTITY_FIELDS: dict[str, str] = {
    "paper_id": "paper_ids",
    "example_id": "example_ids",
    "source_id": "source_ids",
    "source_rollout_id": "source_ids",
    "candidate_id": "candidate_ids",
    "rollout_id": "rollout_ids",
    "session_id": "session_ids",
    "trace_id": "trace_ids",
    "source_trace_id": "trace_ids",
    "address": "addresses",
    "address_id": "addresses",
    "address_sha256": "addresses",
    "seed": "seeds",
    "master_seed": "seeds",
    "scientific_group_id": "groups",
    "group_id": "groups",
    "slot_id": "slots",
    "rollout_slot": "slots",
    "call_id": "call_ids",
    "request_id": "call_ids",
    "decision_id": "call_ids",
    "lineage": "call_ids",
    "node_id": "call_ids",
    "invocation_id": "call_ids",
    "cache_salt": "cache_salts",
    "row_sha256": "row_hashes",
    "canonical_row_sha256": "row_hashes",
    "reference_span": "reference_spans",
    "reference_evidence": "reference_spans",
}
_HISTORICAL_IDENTITY_SET_NAMES = (
    "paper_ids",
    "example_ids",
    "reference_spans",
    "source_ids",
    "candidate_ids",
    "rollout_ids",
    "session_ids",
    "trace_ids",
    "addresses",
    "row_hashes",
    "seeds",
    "groups",
    "slots",
    "cache_salts",
    "call_ids",
)
_ADDRESS_RECORD_FIELDS = (
    "paper_id",
    "example_id",
    "seed",
    "scientific_group_id",
    "slot_id",
    "rollout_slot",
    "cache_salt",
    "row_sha256",
    "canonical_row_sha256",
)


def _collect_historical_identity_values(
    value: object,
    identity_sets: dict[str, set[str]],
    inherited_category: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            category = _HISTORICAL_IDENTITY_FIELDS.get(str(key), inherited_category)
            _collect_historical_identity_values(child, identity_sets, category)
    elif isinstance(value, list):
        for child in value:
            _collect_historical_identity_values(child, identity_sets, inherited_category)
    elif inherited_category is not None and isinstance(value, (str, int)):
        identity_sets[inherited_category].add(str(value))


def _collect_address_records(
    value: object,
    artifact: str,
    role: str,
    records: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        required = {"example_id", "seed", "scientific_group_id", "slot_id", "cache_salt"}
        if required.issubset(value):
            core = {
                field: value[field]
                for field in _ADDRESS_RECORD_FIELDS
                if field in value
            }
            records.append(
                {
                    "artifact": artifact,
                    "role": role,
                    "address_sha256": sha256_json(core),
                    "components": core,
                }
            )
        for child in value.values():
            _collect_address_records(child, artifact, role, records)
    elif isinstance(value, list):
        for child in value:
            _collect_address_records(child, artifact, role, records)


def historical_identity_witness(root: Path) -> dict[str, Any]:
    """Authenticate and enumerate all pre-v13 historical identity witnesses."""

    artifact_hashes: dict[str, str] = {}
    identity_sets = {name: set[str]() for name in _HISTORICAL_IDENTITY_SET_NAMES}
    address_records: list[dict[str, Any]] = []
    retired_records: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []

    for relative, expected in {**HISTORICAL_ADDRESS_HASHES, **HISTORICAL_ROLLOUT_HASHES}.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required historical identity input is missing: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"historical identity input hash mismatch for {relative}")
        artifact_hashes[relative] = digest

    for relative in HISTORICAL_ADDRESS_HASHES:
        report = read_json(root, relative)
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks or not all(
            value is True for value in checks.values()
        ):
            raise ValueError(f"historical address audit is not authenticated: {relative}")
        retired = report.get("retired")
        preserved = report.get("preserved")
        if not isinstance(retired, Mapping) or not isinstance(preserved, list):
            raise ValueError(f"historical address schema is incomplete: {relative}")
        before = len(address_records)
        _collect_address_records(retired, relative, "retired", address_records)
        _collect_address_records(preserved, relative, "preserved", address_records)
        if len(address_records) == before:
            raise ValueError(f"historical address records are empty: {relative}")
        retired_records.append(
            {
                "artifact": relative,
                "components": {
                    field: retired[field] for field in _ADDRESS_RECORD_FIELDS if field in retired
                },
                "address_sha256": sha256_json(
                    {field: retired[field] for field in _ADDRESS_RECORD_FIELDS if field in retired}
                ),
            }
        )
        _collect_historical_identity_values(report, identity_sets)

    for relative in HISTORICAL_ROLLOUT_HASHES:
        report = read_json(root, relative)
        _collect_historical_identity_values(report, identity_sets)
        if relative.endswith("-terminal.json"):
            episode = report.get("first_episode") or report.get("observed_episode")
            if not isinstance(episode, Mapping) or not episode.get("rollout_id"):
                raise ValueError(f"historical rollout identity is missing: {relative}")
            rollout_records.append(
                {
                    "artifact": relative,
                    "rollout_id": str(episode["rollout_id"]),
                    "group_id": str(episode.get("group_id")),
                }
            )

    if len(retired_records) != 6 or len(rollout_records) != 3:
        raise ValueError(
            "historical identity witness does not contain the required v1-v6/v8-v10 set"
        )
    for record in address_records:
        identity_sets["addresses"].add(str(record["address_sha256"]))
    if not identity_sets["rollout_ids"]:
        raise ValueError("historical rollout identity set is empty")

    normalized_sets = {name: sorted(values) for name, values in identity_sets.items()}
    witness = {
        "schema_version": 1,
        "artifacts": artifact_hashes,
        "retired_address_records": retired_records,
        "address_records": address_records,
        "rollout_records": rollout_records,
        "identity_sets": normalized_sets,
    }
    witness["witness_sha256"] = sha256_json(witness)
    return witness


def authenticate_immutable_inputs(root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in FROZEN_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required immutable input is missing: {relative}")
        digest = sha256_file(path)
        actual[relative] = digest
        if digest != expected:
            raise ValueError(
                f"immutable input hash mismatch for {relative}: {digest} != {expected}"
            )
    for relative, expected in POST_REPAIR_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required post-repair artifact is missing: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"post-repair artifact hash mismatch for {relative}")
    source_digest = sha256_file(root / "src/redco/analysis/stage_d_source_producer.py")
    if source_digest != REPAIRED_SOURCE_SHA256:
        raise ValueError("approved repaired producer source hash is not authenticated")
    return actual


def verify_repair_ancestor(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REPAIR_COMMIT, head],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return require_repair_ancestor(
        REPAIR_COMMIT,
        head,
        result.returncode == 0,
    )


def require_repair_ancestor(
    required_commit: str,
    head: str,
    is_ancestor: bool,
) -> str:
    if not is_ancestor:
        raise ValueError(
            f"required repair commit {required_commit} is not an ancestor of HEAD {head}"
        )
    return required_commit


def _safe_extract(archive_bytes: bytes, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError("git archive contains an escaping path") from error
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("git archive member cannot be read")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                target.chmod(member.mode & 0o7777)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(member.linkname, target)
            else:
                raise ValueError(f"unsupported git archive member: {member.name}")


def _gitlink_paths(repository: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--stage"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        mode, _object_id, _stage = metadata.split()
        if mode == "160000":
            paths.append(path)
    return tuple(sorted(paths))


def verify_clean_reconstruction(actual_tree_sha256: str, expected_tree_sha256: str) -> None:
    if actual_tree_sha256 != expected_tree_sha256:
        raise ValueError(
            "clean dependency reconstruction differs: "
            f"{actual_tree_sha256} != {expected_tree_sha256}"
        )


def verify_clean_reconstruction_status(status: str, ignored_probe: str) -> None:
    """Reject files not explained by the authenticated patch stack."""

    if any(line.startswith("??") for line in status.splitlines()):
        raise ValueError("clean dependency reconstruction contains untracked files")
    if ignored_probe.strip():
        raise ValueError("clean dependency reconstruction contains ignored files")


def _reconstruct_prime(root: Path, prime: dict[str, Any], patch_hashes: dict[str, str]) -> str:
    repository = root / "external/prime-rl"
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-prime-reconstruct-") as temporary:
        destination = Path(temporary)
        archive = subprocess.run(
            ["git", "archive", "--format=tar", str(prime["base_commit"])],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        _safe_extract(archive, destination)
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        subprocess.run(["git", "init", "-q"], cwd=destination, check=True)
        subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=redco-draft",
                "-c",
                "user.email=redco-draft@example.invalid",
                "commit",
                "-qm",
                "authenticated base",
            ],
            cwd=destination,
            check=True,
        )
        for patch_name, expected in patch_hashes.items():
            patch = root / "patches" / patch_name
            if sha256_file(patch) != expected:
                raise ValueError(f"pinned dependency patch changed: {patch_name}")
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=destination, check=True)
            subprocess.run(["git", "apply", "--index", str(patch)], cwd=destination, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        clean_probe = subprocess.run(
            ["git", "clean", "-ndx"],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        verify_clean_reconstruction_status(status, clean_probe)
        subprocess.run(["git", "diff", "--check"], cwd=destination, check=True)
        return sha256_bytes(
            canonical_tree_manifest_bytes(
                destination,
                allow_relative_symlinks=True,
                excluded_roots=_gitlink_paths(repository),
            )
        )


def dependency_binding(root: Path, dependency: dict[str, Any]) -> dict[str, Any]:
    prime = next(
        component for component in dependency["components"] if component["name"] == "prime-rl"
    )
    patch_hashes: dict[str, str] = {}
    for patch in prime["patches"]:
        patch_path = root / f"patches/{patch['name']}"
        if not patch_path.is_file():
            raise FileNotFoundError(f"pinned dependency patch missing: {patch['name']}")
        patch_hashes[patch["name"]] = sha256_file(patch_path)
        if patch_hashes[patch["name"]] != patch["sha256"]:
            raise ValueError(f"pinned dependency patch changed: {patch['name']}")
    repository = root / "external/prime-rl"
    gitlink = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if gitlink != prime["base_commit"]:
        raise ValueError("external/prime-rl gitlink differs from the pinned base commit")
    reconstructed = _reconstruct_prime(root, prime, patch_hashes)
    verify_clean_reconstruction(reconstructed, prime["post_tree_sha256"])
    return {
        "gitlink_commit": gitlink,
        "base_commit": prime["base_commit"],
        "expected_patch_sha256": patch_hashes,
        "expected_patch_sequence_sha256": sha256_json(prime["patches"]),
        "expected_post_tree_sha256": prime["post_tree_sha256"],
        "observed_reconstructed_post_tree_sha256": reconstructed,
        "clean_reconstruction_status": "pass",
        "shared_dirty_worktree_not_authoritative": True,
        "verification": (
            "clean git archive at the pinned base plus exact patch bytes was reconstructed "
            "in a disposable tree; the shared dirty worktree, including untracked nested files, "
            "was not used as evidence"
        ),
        "launch_gate": "require the same clean reconstruction and post-tree hash at launch",
    }


def archive_has_evaluator_payload(root: Path) -> dict[str, Any]:
    found: list[str] = []
    archive = root / "runs/stage-d/stage-d1-support-v12-terminal.tar.gz"
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            raw = stream.extractfile(member)
            if raw is None:
                continue
            if any(b"exact_span_f1" in line for line in raw.read().splitlines()):
                found.append(member.name)
    if found != ["stage-d1-support-v12/source-eval/traces.jsonl"]:
        raise ValueError("v12 evaluator payload location is not authenticated")
    return {
        "member": found[0],
        "exact_span_f1": 0.0,
        "info_score_f1": 0.0,
        "classification": "observed_engineering_information_not_admissible_scientific_outcome",
    }


__all__ = [
    "FROZEN_HASHES",
    "HISTORICAL_ADDRESS_HASHES",
    "HISTORICAL_ROLLOUT_HASHES",
    "POST_REPAIR_HASHES",
    "PRE_REPAIR_SOURCE_SHA256",
    "REPAIRED_SOURCE_SHA256",
    "REPAIR_COMMIT",
    "V12_ARCHIVE_SHA256",
    "V12_AUDIT_REPORT_SHA256",
    "V12_DEPENDENCY_STACK_SHA256",
    "V12_EVIDENCE_MANIFEST_SHA256",
    "V12_PLAN_SHA256",
    "V12_PREREG_SHA256",
    "V12_PROTOCOL_SHA256",
    "V12_SOURCE_EVAL_SHA256",
    "V12_SOURCE_SHA256",
    "V12_TERMINAL_REPORT_SHA256",
    "archive_has_evaluator_payload",
    "authenticate_immutable_inputs",
    "dependency_binding",
    "historical_identity_witness",
    "read_json",
    "require_repair_ancestor",
    "sha256_file",
    "verify_clean_reconstruction",
    "verify_clean_reconstruction_status",
    "verify_repair_ancestor",
]
