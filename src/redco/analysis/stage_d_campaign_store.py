"""Atomic compact persistence for one sealed Stage-D three-arm transaction."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from redco.analysis.stage_d_campaign_controller import SealedStageDCampaign
from redco.analysis.stage_d_collection import StageDCollectionPlan
from redco.analysis.stage_d_objective_binding import ArmName, ObjectiveBinding
from redco.analysis.stage_d_three_arm_prime import materialize_prime_rollout_bytes
from redco.contracts import canonical_json

_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class StageDCampaignBundle:
    root: Path
    manifest_bytes: bytes
    manifest_sha256: str
    prime_rollout_paths: tuple[tuple[ArmName, Path], ...]


class StageDCampaignStore:
    """Install or verify a complete campaign bundle without overwriting evidence."""

    def __init__(self, root: Path) -> None:
        if root.exists():
            raise FileExistsError(f"campaign bundle already exists: {root}")
        if not root.name:
            raise ValueError("campaign bundle root must have a terminal name")
        self._root = root

    def persist(
        self,
        *,
        campaign: SealedStageDCampaign,
        ledger_root: Path,
        collection_plan: StageDCollectionPlan,
        collection_receipt_bytes: bytes,
        source_rollout_bytes: Sequence[bytes],
        branch_artifact_bytes: Sequence[bytes],
        objective_binding_bytes: Mapping[ArmName, bytes],
        trainer_toml_bytes: Mapping[ArmName, bytes],
        frozen_inputs: Mapping[str, bytes],
    ) -> StageDCampaignBundle:
        if set(objective_binding_bytes) != set(_ARMS):
            raise ValueError("campaign bundle requires all objective bindings")
        if set(trainer_toml_bytes) != set(_ARMS):
            raise ValueError("campaign bundle requires all trainer TOMLs")
        if collection_plan.to_bytes() == b"" or not collection_receipt_bytes:
            raise ValueError("campaign collection evidence must be nonempty")
        parent = self._root.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{self._root.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        temporary.mkdir(mode=0o700)
        try:
            files: dict[str, bytes] = {}
            self._add(files, "collection/plan.json", collection_plan.to_bytes())
            self._add(files, "collection/receipt.json", collection_receipt_bytes)
            for value in source_rollout_bytes:
                self._add(files, f"sources/{_sha256(value)}.json", value)
            for value in branch_artifact_bytes:
                self._add(files, f"branches/{_sha256(value)}.json", value)
            self._add(
                files,
                "objectives/authorization.json",
                campaign.objective_authorization,
            )
            self._add(files, "ledger/seal.json", campaign.ledger_seal_bytes)
            receipt_by_arm = dict(campaign.batch_authorization_receipts)
            batch_by_arm = {
                "stock": campaign.compilation.stock,
                "branch-global": campaign.compilation.branch_global,
                "local": campaign.compilation.local,
            }
            rollout_paths: list[tuple[ArmName, Path]] = []
            for arm in _ARMS:
                binding_bytes = objective_binding_bytes[arm]
                binding = ObjectiveBinding.from_bytes(binding_bytes)
                batch = batch_by_arm[arm]
                if binding != batch.objective_binding:
                    raise ValueError(f"{arm} objective binding differs from sealed batch")
                self._add(files, f"objectives/{arm}.json", binding_bytes)
                self._add(files, f"objectives/{arm}.toml", trainer_toml_bytes[arm])
                self._add(files, f"batches/{arm}.json", batch.to_bytes())
                self._add(files, f"authorizations/{arm}.json", receipt_by_arm[arm])
                rollout_rel = (
                    f"prime/{arm}/run_default/rollouts/step_{batch.trainer_step}/"
                    "train_rollouts.bin"
                )
                self._add(files, rollout_rel, materialize_prime_rollout_bytes(batch))
                rollout_paths.append((arm, self._root / PurePosixPath(rollout_rel)))
            for name, value in frozen_inputs.items():
                relative = self._safe_relative(name)
                self._add(files, f"frozen/{relative}", value)
            for path in sorted(ledger_root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(ledger_root).as_posix()
                    self._add(
                        files,
                        f"ledger/root/{self._safe_relative(relative)}",
                        path.read_bytes(),
                    )
            entries = []
            for relative, value in sorted(files.items()):
                destination = temporary / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _exclusive_file(destination, value)
                entries.append(
                    {"path": relative, "sha256": _sha256(value), "size_bytes": len(value)}
                )
            manifest = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-sealed-campaign-bundle-v1",
                    "ledger_seal_sha256": campaign.ledger_seal_sha256,
                    "collection_plan_sha256": collection_plan.plan_sha256,
                    "arm_order": list(_ARMS),
                    "entries": entries,
                }
            )
            _exclusive_file(temporary / "manifest.json", manifest)
            _fsync_tree(temporary)
            if self._root.exists():
                raise FileExistsError(f"campaign bundle appeared during install: {self._root}")
            os.replace(temporary, self._root)
            _fsync_directory(parent)
            verified = verify_campaign_bundle(self._root)
            if verified.manifest_bytes != manifest:
                raise ValueError("installed campaign manifest changed after rename")
            return StageDCampaignBundle(
                self._root,
                manifest,
                _sha256(manifest),
                tuple(rollout_paths),
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _add(files: dict[str, bytes], relative: str, value: bytes) -> None:
        if type(value) is not bytes or not value:
            raise ValueError(f"campaign evidence {relative} must be nonempty bytes")
        normalized = StageDCampaignStore._safe_relative(relative)
        previous = files.setdefault(normalized, value)
        if previous != value:
            raise ValueError(f"campaign evidence collision at {normalized}")

    @staticmethod
    def _safe_relative(value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("campaign evidence path must be a safe relative path")
        return path.as_posix()


def verify_campaign_bundle(root: Path) -> StageDCampaignBundle:
    manifest_path = root / "manifest.json"
    manifest = manifest_path.read_bytes()
    import json

    payload = json.loads(manifest)
    if not isinstance(payload, dict) or canonical_json(payload) != manifest:
        raise ValueError("campaign manifest must be canonical JSON")
    if set(payload) != {
        "schema_version",
        "domain",
        "ledger_seal_sha256",
        "collection_plan_sha256",
        "arm_order",
        "entries",
    }:
        raise ValueError("campaign manifest fields differ")
    if (
        payload["schema_version"] != 1
        or payload["domain"] != "redco-stage-d-sealed-campaign-bundle-v1"
        or payload["arm_order"] != list(_ARMS)
    ):
        raise ValueError("unsupported campaign manifest")
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("campaign manifest has no entries")
    expected_paths = {"manifest.json"}
    rollout_paths: list[tuple[ArmName, Path]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("campaign manifest entry fields differ")
        relative = StageDCampaignStore._safe_relative(entry["path"])
        path = root / PurePosixPath(relative)
        value = path.read_bytes()
        if entry["size_bytes"] != len(value) or entry["sha256"] != _sha256(value):
            raise ValueError(f"campaign evidence changed: {relative}")
        expected_paths.add(relative)
    observed_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_paths != expected_paths:
        raise ValueError("campaign bundle contains unmanifested or missing files")
    for arm in _ARMS:
        matches = tuple(
            (root / "prime" / arm).glob(
                "run_default/rollouts/step_*/train_rollouts.bin"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"campaign bundle lacks one exact Prime payload for {arm}")
        rollout_paths.append((arm, matches[0]))
    return StageDCampaignBundle(root, manifest, _sha256(manifest), tuple(rollout_paths))


def _exclusive_file(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_tree(root: Path) -> None:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
