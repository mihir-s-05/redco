"""Atomic compact persistence for one sealed Stage-D three-arm transaction."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from redco.analysis.stage_d_arm_contracts import SealedArmBatch
from redco.analysis.stage_d_campaign_controller import SealedStageDCampaign
from redco.analysis.stage_d_collection import StageDCollectionPlan
from redco.analysis.stage_d_objective_binding import (
    ArmName,
    ObjectiveAuthorization,
    ObjectiveBinding,
)
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_receipt_ledger import (
    LedgerSeal,
    SealedReceiptVerifier,
    inspect_ledger,
)
from redco.analysis.stage_d_shared_initialization import (
    StageDSharedInitializationManifest,
)
from redco.analysis.stage_d_three_arm_prime import (
    _verify_stage_d_batch_authorization,
    materialize_prime_rollout_bytes,
)
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


@dataclass(frozen=True, slots=True)
class FrozenProtocolInputs:
    preregistration: bytes
    dependency_stack_manifest: bytes
    genesis_config: bytes
    source: bytes
    runtime: bytes
    source_eval_config: bytes
    scientific_eval_config: bytes
    heldout_eval_config: bytes
    shared_initialization_manifest: bytes
    base_model_manifest: bytes
    adapter_manifest: bytes | None
    tokenizer_manifest: bytes
    renderer_manifest: bytes
    sampler_conformance_manifest: bytes
    resolved_agent_sampling_law: bytes
    resolved_train_client: bytes

    def files(self, protocol: StageDProtocolManifest) -> dict[str, bytes]:
        StageDSharedInitializationManifest.from_bytes(
            self.shared_initialization_manifest
        ).verify_protocol(protocol)
        values = {
            "protocol/preregistration.json": (
                self.preregistration,
                protocol.preregistration_sha256,
            ),
            "protocol/dependency-stack.json": (
                self.dependency_stack_manifest,
                protocol.dependency_stack_sha256,
            ),
            "protocol/genesis-config.json": (
                self.genesis_config,
                protocol.genesis_config_sha256,
            ),
            "protocol/source.json": (self.source, protocol.source_sha256),
            "protocol/runtime.json": (self.runtime, protocol.runtime_sha256),
            "protocol/source-eval.toml": (
                self.source_eval_config,
                protocol.source_eval_config_sha256,
            ),
            "protocol/scientific-eval.toml": (
                self.scientific_eval_config,
                protocol.scientific_eval_config_sha256,
            ),
            "protocol/heldout-eval.toml": (
                self.heldout_eval_config,
                protocol.heldout_eval_config_sha256,
            ),
            "protocol/shared-initialization.json": (
                self.shared_initialization_manifest,
                protocol.shared_initialization_sha256,
            ),
            "policy/base-model-manifest.json": (
                self.base_model_manifest,
                protocol.policy_identity.base_model_manifest_sha256,
            ),
            "policy/tokenizer-manifest.json": (
                self.tokenizer_manifest,
                protocol.policy_identity.tokenizer_manifest_sha256,
            ),
            "policy/renderer-manifest.json": (
                self.renderer_manifest,
                protocol.policy_identity.renderer_manifest_sha256,
            ),
            "policy/sampler-conformance-manifest.json": (
                self.sampler_conformance_manifest,
                protocol.policy_identity.sampler_conformance_manifest_sha256,
            ),
            "policy/resolved-agent-sampling-law.json": (
                self.resolved_agent_sampling_law,
                protocol.policy_identity.resolved_agent_sampling_law_sha256,
            ),
            "policy/resolved-train-client.json": (
                self.resolved_train_client,
                protocol.policy_identity.resolved_train_client_sha256,
            ),
        }
        adapter_expected = protocol.policy_identity.adapter_manifest_sha256
        if (self.adapter_manifest is None) != (adapter_expected is None):
            raise ValueError("adapter manifest presence differs from protocol")
        if self.adapter_manifest is not None:
            assert adapter_expected is not None
            values["policy/adapter-manifest.json"] = (
                self.adapter_manifest,
                adapter_expected,
            )
        for path, (value, expected) in values.items():
            if type(value) is not bytes or not value or _sha256(value) != expected:
                raise ValueError(f"frozen protocol input differs: {path}")
        return {path: value for path, (value, _) in values.items()}


class StageDCampaignStore:
    """Install or verify a complete campaign bundle without overwriting evidence."""

    def __init__(self, root: Path) -> None:
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
        evaluation_plan_bytes: bytes,
        decision_rule_bytes: bytes,
        reload_probe_bytes: bytes,
        frozen_inputs: FrozenProtocolInputs,
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
            protocol = StageDProtocolManifest.from_bytes(campaign.protocol_manifest)
            if protocol.manifest_sha256 != campaign.protocol_manifest_sha256:
                raise ValueError("sealed campaign protocol identity changed")
            if _sha256(evaluation_plan_bytes) != protocol.evaluation_plan_sha256:
                raise ValueError("evaluation plan differs from protocol")
            if _sha256(decision_rule_bytes) != protocol.decision_rule_sha256:
                raise ValueError("decision rule differs from protocol")
            if _sha256(reload_probe_bytes) != protocol.reload_probe_sha256:
                raise ValueError("reload probe differs from protocol")
            self._add(files, "protocol/manifest.json", campaign.protocol_manifest)
            for relative, value in frozen_inputs.files(protocol).items():
                self._add(files, relative, value)
            self._add(files, "evaluation/plan.json", evaluation_plan_bytes)
            self._add(files, "evaluation/decision-rule.json", decision_rule_bytes)
            self._add(files, "evaluation/reload-probe.json", reload_probe_bytes)
            self._add(files, "collection/plan.json", collection_plan.to_bytes())
            self._add(files, "collection/receipt.json", collection_receipt_bytes)
            for value in source_rollout_bytes:
                self._add(files, f"sources/{_source_semantic_sha256(value)}.json", value)
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
            scan = inspect_ledger(ledger_root)
            if scan.status != "sealed-valid" or scan.seal != campaign.ledger_seal:
                raise ValueError("campaign ledger root differs from its terminal seal")
            for path in sorted((ledger_root / "records").glob("*.json")):
                self._add(files, f"ledger/root/records/{path.name}", path.read_bytes())
            for digest in sorted(scan.evidence_refs):
                self._add(
                    files,
                    f"ledger/root/evidence/{digest}",
                    (ledger_root / "evidence" / digest).read_bytes(),
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
                    "protocol_manifest_sha256": campaign.protocol_manifest_sha256,
                    "collection_plan_sha256": collection_plan.plan_sha256,
                    "arm_order": list(_ARMS),
                    "entries": entries,
                }
            )
            _exclusive_file(temporary / "manifest.json", manifest)
            _fsync_tree(temporary)
            if self._root.exists():
                return _require_existing_bundle(
                    self._root,
                    expected_manifest=manifest,
                    expected_files=files,
                )
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
    payload = json.loads(manifest)
    if not isinstance(payload, dict) or canonical_json(payload) != manifest:
        raise ValueError("campaign manifest must be canonical JSON")
    if set(payload) != {
        "schema_version",
        "domain",
        "ledger_seal_sha256",
        "protocol_manifest_sha256",
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
    required = {
        "protocol/manifest.json",
        "protocol/preregistration.json",
        "protocol/dependency-stack.json",
        "protocol/genesis-config.json",
        "protocol/source.json",
        "protocol/runtime.json",
        "protocol/source-eval.toml",
        "protocol/scientific-eval.toml",
        "protocol/heldout-eval.toml",
        "protocol/shared-initialization.json",
        "policy/base-model-manifest.json",
        "policy/tokenizer-manifest.json",
        "policy/renderer-manifest.json",
        "policy/sampler-conformance-manifest.json",
        "policy/resolved-agent-sampling-law.json",
        "policy/resolved-train-client.json",
        "evaluation/plan.json",
        "evaluation/decision-rule.json",
        "evaluation/reload-probe.json",
        "collection/plan.json",
        "collection/receipt.json",
        "objectives/authorization.json",
        "ledger/seal.json",
    }
    for arm in _ARMS:
        required.update(
            {
                f"objectives/{arm}.json",
                f"objectives/{arm}.toml",
                f"batches/{arm}.json",
                f"authorizations/{arm}.json",
            }
        )
    if not required.issubset(observed_paths):
        raise ValueError("campaign bundle lacks required semantic inputs")
    protocol_bytes = (root / "protocol" / "manifest.json").read_bytes()
    protocol = StageDProtocolManifest.from_bytes(protocol_bytes)
    if protocol.manifest_sha256 != payload["protocol_manifest_sha256"]:
        raise ValueError("campaign protocol manifest link differs")
    StageDSharedInitializationManifest.from_bytes(
        (root / "protocol" / "shared-initialization.json").read_bytes()
    ).verify_protocol(protocol)
    frozen_hashes = {
        "protocol/preregistration.json": protocol.preregistration_sha256,
        "protocol/dependency-stack.json": protocol.dependency_stack_sha256,
        "protocol/genesis-config.json": protocol.genesis_config_sha256,
        "protocol/source.json": protocol.source_sha256,
        "protocol/runtime.json": protocol.runtime_sha256,
        "protocol/source-eval.toml": protocol.source_eval_config_sha256,
        "protocol/scientific-eval.toml": protocol.scientific_eval_config_sha256,
        "protocol/heldout-eval.toml": protocol.heldout_eval_config_sha256,
        "protocol/shared-initialization.json": protocol.shared_initialization_sha256,
        "policy/base-model-manifest.json": (
            protocol.policy_identity.base_model_manifest_sha256
        ),
        "policy/tokenizer-manifest.json": (
            protocol.policy_identity.tokenizer_manifest_sha256
        ),
        "policy/renderer-manifest.json": (
            protocol.policy_identity.renderer_manifest_sha256
        ),
        "policy/sampler-conformance-manifest.json": (
            protocol.policy_identity.sampler_conformance_manifest_sha256
        ),
        "policy/resolved-agent-sampling-law.json": (
            protocol.policy_identity.resolved_agent_sampling_law_sha256
        ),
        "policy/resolved-train-client.json": (
            protocol.policy_identity.resolved_train_client_sha256
        ),
    }
    adapter_sha256 = protocol.policy_identity.adapter_manifest_sha256
    if adapter_sha256 is not None:
        required.add("policy/adapter-manifest.json")
        frozen_hashes["policy/adapter-manifest.json"] = adapter_sha256
    elif "policy/adapter-manifest.json" in observed_paths:
        raise ValueError("campaign contains an unexpected adapter manifest")
    for relative, expected in frozen_hashes.items():
        if _sha256((root / PurePosixPath(relative)).read_bytes()) != expected:
            raise ValueError(f"campaign frozen input differs from protocol: {relative}")
    if _sha256((root / "evaluation" / "plan.json").read_bytes()) != (
        protocol.evaluation_plan_sha256
    ):
        raise ValueError("campaign evaluation plan differs from protocol")
    if _sha256((root / "evaluation" / "decision-rule.json").read_bytes()) != (
        protocol.decision_rule_sha256
    ):
        raise ValueError("campaign decision rule differs from protocol")
    if _sha256((root / "evaluation" / "reload-probe.json").read_bytes()) != (
        protocol.reload_probe_sha256
    ):
        raise ValueError("campaign reload probe differs from protocol")
    plan = StageDCollectionPlan.from_bytes((root / "collection" / "plan.json").read_bytes())
    if plan.plan_sha256 != payload["collection_plan_sha256"] or (
        plan.plan_sha256 != protocol.collection_plan_sha256
    ):
        raise ValueError("campaign collection plan link differs")
    seal_bytes = (root / "ledger" / "seal.json").read_bytes()
    if _sha256(seal_bytes) != payload["ledger_seal_sha256"]:
        raise ValueError("campaign ledger seal link differs")
    seal = LedgerSeal.from_bytes(seal_bytes)
    verifier = SealedReceiptVerifier(root / "ledger" / "root", seal)
    authorization_bytes = (root / "objectives" / "authorization.json").read_bytes()
    if _sha256(authorization_bytes) != protocol.objective_authorization_sha256:
        raise ValueError("campaign objective authorization differs from protocol")
    authorization = ObjectiveAuthorization.from_bytes(authorization_bytes)
    bindings: list[ObjectiveBinding] = []
    batches: dict[ArmName, SealedArmBatch] = {}
    collection_receipt_sha256 = _sha256(
        (root / "collection" / "receipt.json").read_bytes()
    )
    for arm in _ARMS:
        binding_bytes = (root / "objectives" / f"{arm}.json").read_bytes()
        trainer_bytes = (root / "objectives" / f"{arm}.toml").read_bytes()
        if _sha256(binding_bytes) != protocol.arm_hash("objective_binding", arm):
            raise ValueError(f"campaign {arm} objective differs from protocol")
        if _sha256(trainer_bytes) != protocol.arm_hash("trainer_config", arm):
            raise ValueError(f"campaign {arm} trainer differs from protocol")
        binding = ObjectiveBinding.from_bytes(binding_bytes)
        if binding.arm != arm:
            raise ValueError(f"campaign {arm} objective names another arm")
        bindings.append(binding)
        batch = SealedArmBatch.verify_bytes((root / "batches" / f"{arm}.json").read_bytes())
        batches[arm] = batch
        if (
            batch.arm != arm
            or batch.objective_binding != binding
            or batch.trainer_step != protocol.trainer_step
            or batch.seq_len != protocol.seq_len
        ):
            raise ValueError(f"campaign {arm} batch differs from protocol inputs")
        receipt = (root / "authorizations" / f"{arm}.json").read_bytes()
        _verify_stage_d_batch_authorization(
            receipt,
            verifier=verifier,
            batch=batch,
            sealed_batch_bytes=batch.to_bytes(),
            objective_authorization_sha256=protocol.objective_authorization_sha256,
        )
        anchored = verifier(
            receipt,
            receipt_kind="stage_d_training_batch_authorization",
        )
        if anchored.get("collection_receipt_sha256") != collection_receipt_sha256:
            raise ValueError(f"campaign {arm} authorization names another collection")
        matches = tuple(
            (root / "prime" / arm).glob(
                "run_default/rollouts/step_*/train_rollouts.bin"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"campaign bundle lacks one exact Prime payload for {arm}")
        if matches[0].read_bytes() != materialize_prime_rollout_bytes(batch):
            raise ValueError(f"campaign {arm} Prime payload differs from its sealed batch")
        rollout_paths.append((arm, matches[0]))
    authorization.authorize(tuple(bindings))
    source_roster = batches["stock"].source_sha256s
    if any(batch.source_sha256s != source_roster for batch in batches.values()):
        raise ValueError("campaign batches differ in source roster")
    observed_sources = tuple(
        sorted(path.stem for path in (root / "sources").glob("*.json"))
    )
    if observed_sources != source_roster:
        raise ValueError("campaign source files differ from the sealed source roster")
    for path in (root / "sources").glob("*.json"):
        if _verify_bundled_source(path.read_bytes(), verifier=verifier) != path.stem:
            raise ValueError("campaign source filename differs from verified source bytes")
    branch_roster = batches["local"].branch_artifact_sha256s
    if batches["branch-global"].branch_artifact_sha256s != branch_roster:
        raise ValueError("campaign branch arms differ in artifact roster")
    observed_branches = tuple(
        sorted(path.stem for path in (root / "branches").glob("*.json"))
    )
    if observed_branches != branch_roster:
        raise ValueError("campaign branch files differ from the sealed branch roster")
    if any(
        _sha256(path.read_bytes()) != path.stem
        for path in (root / "branches").glob("*.json")
    ):
        raise ValueError("campaign branch filename differs from its raw digest")
    _verify_bundled_collection_receipt(
        plan,
        (root / "collection" / "receipt.json").read_bytes(),
        source_roster,
    )
    return StageDCampaignBundle(root, manifest, _sha256(manifest), tuple(rollout_paths))


def _verify_bundled_collection_receipt(
    plan: StageDCollectionPlan,
    receipt_bytes: bytes,
    source_roster: tuple[str, ...],
) -> None:
    try:
        payload = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bundled collection receipt is not JSON") from error
    if (
        not isinstance(payload, dict)
        or canonical_json(payload) != receipt_bytes
        or set(payload)
        != {
            "schema_version",
            "domain",
            "plan_sha256",
            "planned_slot_count",
            "terminal_slot_count",
            "dispositions",
        }
        or payload.get("schema_version") != 1
        or payload.get("domain") != "redco-stage-d-source-collection-receipt-v1"
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("planned_slot_count") != len(plan.slots)
        or payload.get("terminal_slot_count") != len(plan.slots)
        or not isinstance(payload.get("dispositions"), list)
        or len(payload["dispositions"]) != len(plan.slots)
    ):
        raise ValueError("bundled collection receipt differs from its frozen plan")
    observed: list[str] = []
    expected_fields = {
        "slot_id",
        "example_id",
        "rollout_slot",
        "seed",
        "cache_salt",
        "rollout_id",
        "source_sha256",
        "branch_eligible",
        "disposition",
        "ineligibility_reason",
    }
    for slot, disposition in zip(plan.slots, payload["dispositions"], strict=True):
        if (
            not isinstance(disposition, dict)
            or set(disposition) != expected_fields
            or disposition.get("slot_id") != slot.slot_id
            or disposition.get("example_id") != slot.example_id
            or disposition.get("rollout_slot") != slot.rollout_slot
            or disposition.get("seed") != slot.seed
            or disposition.get("cache_salt") != slot.cache_salt
            or not isinstance(disposition.get("rollout_id"), str)
            or not disposition["rollout_id"]
            or not isinstance(disposition.get("source_sha256"), str)
        ):
            raise ValueError("bundled collection disposition differs from its slot")
        observed.append(disposition["source_sha256"])
    if tuple(sorted(observed)) != source_roster or len(set(observed)) != len(observed):
        raise ValueError("bundled collection receipt differs from source roster")


def _source_semantic_sha256(value: bytes) -> str:
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bundled source is not JSON") from error
    digest = payload.get("source_sha256") if isinstance(payload, dict) else None
    source = payload.get("source") if isinstance(payload, dict) else None
    producer = payload.get("producer_receipt") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or canonical_json(payload) != value
        or payload.get("schema_version") != 1
        or payload.get("domain") != "redco-stage-d-source-rollout-v1"
        or set(payload)
        != {"schema_version", "domain", "source", "source_sha256", "producer_receipt"}
        or not isinstance(source, dict)
        or not isinstance(producer, dict)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("bundled source envelope is invalid")
    recomputed = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-source-rollout-v1",
                "source": source,
            }
        )
    )
    if recomputed != digest or producer.get("source_sha256") != digest:
        raise ValueError("bundled source semantic digest or producer receipt differs")
    return digest


def _verify_bundled_source(
    value: bytes,
    *,
    verifier: SealedReceiptVerifier,
) -> str:
    digest = _source_semantic_sha256(value)
    envelope = json.loads(value)
    source = envelope["source"]
    producer = envelope["producer_receipt"]
    producer_bytes = canonical_json(producer)
    anchored = dict(
        verifier(producer_bytes, receipt_kind="source_rollout_completed")
    )
    if anchored != producer:
        raise ValueError("bundled source producer receipt is not ledger-anchored")
    expected = {
        "group_id": source.get("group_id"),
        "rollout_id": source.get("rollout_id"),
        "source_sha256": digest,
        "trace_sha256": source.get("trace_sha256"),
        "reward_evidence_sha256": source.get("reward_evidence_sha256"),
        "stock_sequences_evidence_sha256": source.get(
            "stock_sequences_evidence_sha256"
        ),
        "base_model_manifest_sha256": source.get("base_model_manifest_sha256"),
    }
    if any(anchored.get(name) != item for name, item in expected.items()):
        raise ValueError("bundled source producer receipt differs from source payload")
    return digest


def _require_existing_bundle(
    root: Path,
    *,
    expected_manifest: bytes,
    expected_files: Mapping[str, bytes],
) -> StageDCampaignBundle:
    existing = verify_campaign_bundle(root)
    if existing.manifest_bytes != expected_manifest:
        raise FileExistsError("existing campaign bundle has different manifest bytes")
    for relative, expected in expected_files.items():
        if (root / PurePosixPath(relative)).read_bytes() != expected:
            raise FileExistsError(f"existing campaign evidence differs: {relative}")
    return existing


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
