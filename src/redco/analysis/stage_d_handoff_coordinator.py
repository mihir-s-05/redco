"""Thin crash-safe adoption journal for the Stage-D production handoff."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from redco.analysis.stage_d_campaign_store import verify_campaign_bundle
from redco.analysis.stage_d_checkpoint_evidence import StageDCheckpointManifest
from redco.analysis.stage_d_checkpoint_materialization import (
    materialize_adopted_checkpoint,
)
from redco.analysis.stage_d_evaluation_barrier import (
    EvaluationCheckpointBinding,
    StageDEvaluationAuthorization,
    StageDSealedEvaluationCompletion,
)
from redco.analysis.stage_d_evaluation_codec import (
    EvaluationEvidenceStore,
    FaultHook,
    atomic_publish,
    decode_record,
    encode_record,
    exclusive_lock,
    sha256,
)
from redco.analysis.stage_d_evaluation_contracts import (
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_provider_billing import StageDProviderBilling
from redco.analysis.stage_d_runtime_bundle import verify_evaluation_runtime_bundle
from redco.analysis.stage_d_terminalization import (
    HandoffCoordinator,
    StageDCleanupEvidence,
    StageDDecisionVector,
    TerminalPhase,
    TerminalStatus,
    TerminationCode,
    finalize_stage_d,
    verify_stage_d_terminal_seal,
)
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.analysis.stage_d_training_completion import StageDTrainingCompletion
from redco.contracts import canonical_json

_EVENT_FIELDS = {
    "genesis": {
        "preregistration_sha256",
        "protocol_manifest_sha256",
        "handoff_policy_sha256",
    },
    "campaign_adopted": {
        "campaign_bundle_manifest_sha256",
        "adoption_manifest_sha256",
    },
    "training_adopted": {
        "training_completion_sha256",
        "adoption_manifest_sha256",
        "trainer_ledger_head_sha256",
        "trainer_record_count",
    },
    "evaluation_authorized": {
        "training_adoption_record_sha256",
        "evaluation_authorization_sha256",
        "execution_manifest_sha256",
        "evaluation_ledger_id",
        "runtime_bundle_sha256",
    },
    "evaluation_adopted": {
        "evaluation_completion_sha256",
        "adoption_manifest_sha256",
        "evaluation_ledger_head_sha256",
        "evaluation_record_count",
    },
    "report_committed": {
        "report_sha256",
        "decision_evidence_sha256",
        "billing_evidence_sha256",
    },
    "seal": {"report_record_sha256"},
}
_ORDER = tuple(_EVENT_FIELDS)
_ADOPTION_DOMAIN = "redco-stage-d-handoff-adoption-manifest-v1"


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("handoff adoption path is unsafe")
    return value


@dataclass(frozen=True, slots=True)
class StageDHandoffSnapshot:
    preregistration_sha256: str
    protocol_manifest_sha256: str
    handoff_policy_sha256: str
    campaign_bundle_manifest_sha256: str | None
    training_completion_sha256: str | None
    evaluation_authorization_record_sha256: str | None
    execution_manifest_sha256: str | None
    evaluation_ledger_id: str | None
    evaluation_authorization_sha256: str | None
    evaluation_completion_sha256: str | None
    report_record_sha256: str | None
    terminal_seal_sha256: str | None
    sealed: bool
    head_sha256: str
    record_count: int
    events: tuple[tuple[str, dict[str, Any], str], ...]

    def event(self, kind: str) -> tuple[dict[str, Any], str] | None:
        matches = [(event, digest) for name, event, digest in self.events if name == kind]
        if len(matches) > 1:
            raise ValueError("handoff event kind was duplicated")
        return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class AuthorizedEvaluation:
    authorization: StageDEvaluationAuthorization
    authorization_bytes: bytes
    authorization_record_sha256: str


class StageDHandoffCoordinator:
    """Adopt immutable subordinate bundles; never supervise their micro-events."""

    def __init__(self, root: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.root = root
        self.records = root / "records"
        self.evidence = EvaluationEvidenceStore(root / "evidence")
        self.lock_path = root / "writer.lock"
        self._fault_hook = fault_hook

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        preregistration_sha256: str,
        protocol_manifest_sha256: str,
        handoff_policy_sha256: str,
        fault_hook: FaultHook | None = None,
    ) -> StageDHandoffCoordinator:
        genesis = {
            "preregistration_sha256": _require_sha256(
                preregistration_sha256, "preregistration sha256"
            ),
            "protocol_manifest_sha256": _require_sha256(
                protocol_manifest_sha256, "protocol manifest sha256"
            ),
            "handoff_policy_sha256": _require_sha256(
                handoff_policy_sha256, "handoff policy sha256"
            ),
        }
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FileExistsError("handoff root is not a regular directory")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if any(
            item.name not in {"records", "evidence", "writer.lock", "terminal_seal.json"}
            for item in root.iterdir()
        ):
            raise FileExistsError("handoff root contains unknown state")
        coordinator = cls(root, fault_hook=fault_hook)
        coordinator.records.mkdir(exist_ok=True)
        coordinator.evidence.root.mkdir(exist_ok=True)
        with exclusive_lock(coordinator.lock_path):
            if not any(coordinator.records.glob("*.json")):
                coordinator._append_unlocked("genesis", genesis)
            snapshot = coordinator.inspect()
            if snapshot.events[0][1] != genesis:
                raise FileExistsError("handoff coordinator has a different genesis")
        return coordinator

    def inspect(self) -> StageDHandoffSnapshot:
        paths = sorted(self.records.glob("*.json"))
        if [path.name for path in paths] != [f"{index:08d}.json" for index in range(len(paths))]:
            raise ValueError("handoff record roster is noncontiguous")
        records = []
        digests: list[str] = []
        for index, path in enumerate(paths):
            if path.is_symlink() or not path.is_file():
                raise ValueError("handoff record is absent or symbolic")
            value = path.read_bytes()
            record = decode_record(value)
            digest = sha256(value)
            expected_prior = None if index == 0 else digests[-1]
            if record["offset"] != index or record["prior_record_sha256"] != expected_prior:
                raise ValueError("handoff record hash chain differs")
            kind = record["record_kind"]
            event = record["event"]
            if kind not in _EVENT_FIELDS or set(event) != _EVENT_FIELDS[kind]:
                raise ValueError("handoff event fields differ")
            records.append(record)
            digests.append(digest)
        if not records or records[0]["record_kind"] != "genesis":
            raise ValueError("handoff coordinator lacks genesis")
        events = tuple(
            (record["record_kind"], record["event"], digest)
            for record, digest in zip(records, digests, strict=True)
        )
        observed_order = tuple(_ORDER.index(kind) for kind, _, _ in events)
        if observed_order != tuple(range(len(events))):
            raise ValueError("handoff events are duplicated or out of order")
        adoption_kinds = {
            "campaign_adopted": "campaign",
            "training_adopted": "training",
            "evaluation_adopted": "evaluation",
        }
        for kind, event, _ in events:
            for name, value in event.items():
                if name.endswith("sha256") and value is not None:
                    _require_sha256(value, f"handoff {name}")
            for name in (
                "adoption_manifest_sha256",
                "execution_manifest_sha256",
                "runtime_bundle_sha256",
                "evaluation_authorization_sha256",
                "evaluation_completion_sha256",
                "report_sha256",
                "decision_evidence_sha256",
                "billing_evidence_sha256",
            ):
                digest = event.get(name)
                if digest is not None:
                    self.evidence.get(digest)
            adoption = event.get("adoption_manifest_sha256")
            if adoption is not None:
                adoption_payload = self._verify_adoption_manifest(self.evidence.get(adoption))
                if adoption_payload["kind"] != adoption_kinds[kind]:
                    raise ValueError("handoff adoption kind differs from its event")
        genesis = events[0][1]
        by_kind = {kind: (event, digest) for kind, event, digest in events}
        campaign = by_kind.get("campaign_adopted")
        training = by_kind.get("training_adopted")
        authorization = by_kind.get("evaluation_authorized")
        evaluation = by_kind.get("evaluation_adopted")
        report = by_kind.get("report_committed")
        if campaign is not None:
            self._verify_copied_campaign(campaign[0], genesis)
        if training is not None:
            self._verify_copied_training(training[0], genesis, campaign)
        if evaluation is not None:
            self._verify_copied_evaluation(evaluation[0], authorization)
        if authorization is not None:
            if (
                training is None
                or authorization[0]["training_adoption_record_sha256"] != training[1]
            ):
                raise ValueError("evaluation authorization differs from training adoption")
            authorization_bytes = self.evidence.get(
                authorization[0]["evaluation_authorization_sha256"]
            )
            authorization_value = StageDEvaluationAuthorization.from_bytes(
                authorization_bytes
            )
            execution_bytes = self.evidence.get(
                authorization[0]["execution_manifest_sha256"]
            )
            execution_value = StageDEvaluationExecutionManifest.from_bytes(execution_bytes)
            if (
                authorization_value.handoff_training_adoption_record_sha256
                != training[1]
                or authorization_value.execution_manifest_sha256
                != authorization[0]["execution_manifest_sha256"]
                or execution_value.manifest_sha256
                != authorization[0]["execution_manifest_sha256"]
                or execution_value.evaluation_ledger_id
                != authorization[0]["evaluation_ledger_id"]
                or execution_value.runtime_bundle_sha256
                != authorization[0]["runtime_bundle_sha256"]
            ):
                raise ValueError("evaluation authorization evidence differs")
        if events[-1][0] == "seal" and (
            report is None or events[-1][1]["report_record_sha256"] != report[1]
        ):
            raise ValueError("handoff seal differs from its report")
        terminal_path = self.root / "terminal_seal.json"
        terminal_sha256 = None
        if terminal_path.exists():
            if terminal_path.is_symlink() or not terminal_path.is_file():
                raise ValueError("handoff terminal seal is absent or symbolic")
            if events[-1][0] == "seal":
                raise ValueError("handoff has both legacy and typed terminal seals")
            terminal_bytes = terminal_path.read_bytes()
            verify_stage_d_terminal_seal(
                cast(HandoffCoordinator, self),
                terminal_bytes,
                preregistration_sha256=genesis["preregistration_sha256"],
                protocol_manifest_sha256=genesis["protocol_manifest_sha256"],
                handoff_policy_sha256=genesis["handoff_policy_sha256"],
                handoff_head_sha256=digests[-1],
                handoff_record_count=len(records),
            )
            terminal_sha256 = sha256(terminal_bytes)
        return StageDHandoffSnapshot(
            genesis["preregistration_sha256"],
            genesis["protocol_manifest_sha256"],
            genesis["handoff_policy_sha256"],
            None if campaign is None else campaign[0]["campaign_bundle_manifest_sha256"],
            None if training is None else training[0]["training_completion_sha256"],
            None if authorization is None else authorization[1],
            None if authorization is None else authorization[0]["execution_manifest_sha256"],
            None if authorization is None else authorization[0]["evaluation_ledger_id"],
            None if authorization is None else authorization[0]["evaluation_authorization_sha256"],
            None if evaluation is None else evaluation[0]["evaluation_completion_sha256"],
            None if report is None else report[1],
            terminal_sha256,
            events[-1][0] == "seal" or terminal_sha256 is not None,
            digests[-1],
            len(records),
            events,
        )

    def adopt_campaign(self, bundle_root: Path) -> str:
        bundle = verify_campaign_bundle(bundle_root)
        payload = json.loads(bundle.manifest_bytes)
        if payload["protocol_manifest_sha256"] != self.inspect().protocol_manifest_sha256:
            raise ValueError("campaign bundle protocol differs from handoff genesis")
        entries = [("manifest.json", bundle.manifest_bytes)]
        for item in payload["entries"]:
            value = (bundle_root / PurePosixPath(item["path"])).read_bytes()
            if sha256(value) != item["sha256"] or len(value) != item["size_bytes"]:
                raise ValueError("campaign changed while it was being copied")
            entries.append((item["path"], value))
        adoption_sha256 = self._install_adoption("campaign", entries)
        event = {
            "campaign_bundle_manifest_sha256": bundle.manifest_sha256,
            "adoption_manifest_sha256": adoption_sha256,
        }
        return self._transition("campaign_adopted", event, requires="genesis")

    def adopt_training(self, ledger: StageDTrainerRunLedger) -> str:
        completion = StageDTrainingCompletion.build(ledger)
        snapshot = self.inspect()
        if (
            completion.protocol_manifest_sha256 != snapshot.protocol_manifest_sha256
            or completion.campaign_manifest_sha256 != snapshot.campaign_bundle_manifest_sha256
        ):
            raise ValueError("training completion differs from adopted campaign")
        entries = [("completion.json", completion.to_bytes())]
        entries.extend(
            (f"records/{path.name}", path.read_bytes())
            for path in sorted(ledger.records.glob("*.json"))
        )
        entries.extend(
            (f"evidence/{path.name}", path.read_bytes())
            for path in sorted(ledger.evidence.iterdir())
            if path.is_file() and not path.is_symlink()
        )
        copied_records = tuple(value for name, value in entries if name.startswith("records/"))
        copied_evidence = {
            name.removeprefix("evidence/"): value
            for name, value in entries
            if name.startswith("evidence/")
        }
        if (
            tuple(sha256(value) for value in copied_records) != completion.record_sha256s
            or tuple(sorted(copied_evidence)) != completion.evidence_sha256s
            or any(sha256(value) != name for name, value in copied_evidence.items())
        ):
            raise ValueError("training changed while it was being copied")
        adoption_sha256 = self._install_adoption("training", entries)
        event = {
            "training_completion_sha256": completion.completion_sha256,
            "adoption_manifest_sha256": adoption_sha256,
            "trainer_ledger_head_sha256": completion.trainer_ledger_head_sha256,
            "trainer_record_count": completion.trainer_record_count,
        }
        return self._transition("training_adopted", event, requires="campaign_adopted")

    def authorize_evaluation(
        self,
        *,
        execution_manifest_bytes: bytes,
        runtime_bundle_bytes: bytes,
    ) -> AuthorizedEvaluation:
        manifest = StageDEvaluationExecutionManifest.from_bytes(execution_manifest_bytes)
        snapshot = self.inspect()
        protocol = self._adopted_protocol()
        training_event = snapshot.event("training_adopted")
        if training_event is None:
            raise RuntimeError("handoff lacks adopted training")
        training = StageDTrainingCompletion.from_bytes(
            self._adopted_entry("training_adopted", "completion.json")
        )
        if (
            manifest.protocol_manifest_sha256 != snapshot.protocol_manifest_sha256
            or manifest.heldout_eval_config_sha256 != protocol.heldout_eval_config_sha256
            or manifest.evaluation_plan_sha256 != protocol.evaluation_plan_sha256
            or manifest.decision_rule_sha256 != protocol.decision_rule_sha256
            or manifest.trainer_ledger_head_sha256
            != training.trainer_ledger_head_sha256
            or manifest.trainer_record_count
            != training.trainer_record_count
            or training.protocol_manifest_sha256 != snapshot.protocol_manifest_sha256
            or training.campaign_manifest_sha256
            != snapshot.campaign_bundle_manifest_sha256
        ):
            raise ValueError("evaluation manifest differs from adopted training")
        expected_checkpoints = tuple(
            EvaluationCheckpointBinding(
                item.arm,
                item.checkpoint_manifest_sha256,
                item.post_model_sha256,
                item.reload_evidence_sha256,
            )
            for item in training.arms
        )
        for checkpoint in expected_checkpoints:
            for role in ("server", "client"):
                program = manifest.program(checkpoint.arm, role)
                if (
                    program.checkpoint_manifest_sha256
                    != checkpoint.checkpoint_manifest_sha256
                    or program.post_model_sha256 != checkpoint.post_model_sha256
                    or program.reload_evidence_sha256 != checkpoint.reload_evidence_sha256
                ):
                    raise ValueError(
                        "evaluation program checkpoint differs from adopted training"
                    )
        if sha256(runtime_bundle_bytes) != manifest.runtime_bundle_sha256:
            raise ValueError("evaluation runtime bundle differs from its manifest")
        verify_evaluation_runtime_bundle(runtime_bundle_bytes, manifest=manifest)
        retained_runtime_path = self.evidence.root / manifest.runtime_bundle_sha256
        if (
            Path(manifest.runtime_bundle_path).is_symlink()
            or Path(manifest.runtime_bundle_path).resolve()
            != retained_runtime_path.resolve()
        ):
            raise ValueError("evaluation runtime path is not coordinator-owned evidence")
        authorization = StageDEvaluationAuthorization(
            handoff_training_adoption_record_sha256=training_event[1],
            campaign_manifest_sha256=self._required_digest(
                snapshot.campaign_bundle_manifest_sha256
            ),
            protocol_manifest_sha256=snapshot.protocol_manifest_sha256,
            trainer_ledger_head_sha256=training.trainer_ledger_head_sha256,
            trainer_record_count=training.trainer_record_count,
            heldout_eval_config_sha256=protocol.heldout_eval_config_sha256,
            evaluation_plan_sha256=protocol.evaluation_plan_sha256,
            execution_manifest_sha256=manifest.manifest_sha256,
            checkpoints=expected_checkpoints,
        )
        authorization_bytes = authorization.to_bytes()
        execution_sha256 = self._put_evidence(execution_manifest_bytes)
        runtime_sha256 = self._put_evidence(runtime_bundle_bytes)
        authorization_sha256 = self._put_evidence(authorization_bytes)
        event = {
            "training_adoption_record_sha256": training_event[1],
            "evaluation_authorization_sha256": authorization_sha256,
            "execution_manifest_sha256": execution_sha256,
            "evaluation_ledger_id": manifest.evaluation_ledger_id,
            "runtime_bundle_sha256": runtime_sha256,
        }
        self._transition("evaluation_authorized", event, requires="training_adopted")
        return self.load_authorized_evaluation()

    def load_authorized_evaluation(self) -> AuthorizedEvaluation:
        snapshot = self.inspect()
        event = snapshot.event("evaluation_authorized")
        if event is None:
            raise RuntimeError("handoff lacks evaluation authorization")
        value = self.evidence.get(event[0]["evaluation_authorization_sha256"])
        return AuthorizedEvaluation(
            StageDEvaluationAuthorization.from_bytes(value),
            value,
            event[1],
        )

    def materialize_evaluation_ledger(self, evaluation_root: Path) -> StageDEvaluationLedger:
        authorized = self.load_authorized_evaluation()
        snapshot = self.inspect()
        manifest_bytes = self.evidence.get(
            self._required_digest(snapshot.execution_manifest_sha256)
        )
        manifest = StageDEvaluationExecutionManifest.from_bytes(manifest_bytes)
        runtime_bytes = self.evidence.get(manifest.runtime_bundle_sha256)
        plan_bytes = self._adopted_entry("campaign_adopted", "evaluation/plan.json")
        if sha256(plan_bytes) != authorized.authorization.evaluation_plan_sha256:
            raise ValueError("adopted evaluation plan differs from authorization")
        return StageDEvaluationLedger.create(
            evaluation_root / manifest.evaluation_ledger_id,
            authorization_bytes=authorized.authorization_bytes,
            execution_manifest_bytes=manifest_bytes,
            evaluation_plan_bytes=plan_bytes,
            runtime_bundle_bytes=runtime_bytes,
        )

    def _adopted_protocol(self) -> StageDProtocolManifest:
        snapshot = self.inspect()
        campaign = snapshot.event("campaign_adopted")
        if campaign is None:
            raise RuntimeError("handoff lacks an adopted campaign")
        adoption = json.loads(self.evidence.get(campaign[0]["adoption_manifest_sha256"]))
        match = [item for item in adoption["entries"] if item["path"] == "protocol/manifest.json"]
        if len(match) != 1:
            raise ValueError("campaign adoption lacks one protocol manifest")
        protocol_bytes = self.evidence.get(match[0]["sha256"])
        protocol = StageDProtocolManifest.from_bytes(protocol_bytes)
        if protocol.manifest_sha256 != snapshot.protocol_manifest_sha256:
            raise ValueError("adopted protocol differs from handoff genesis")
        return protocol

    def _adopted_entry(self, event_kind: str, relative: str) -> bytes:
        event = self.inspect().event(event_kind)
        if event is None:
            raise RuntimeError(f"handoff lacks required {event_kind} event")
        adoption = json.loads(self.evidence.get(event[0]["adoption_manifest_sha256"]))
        matches = [item for item in adoption["entries"] if item["path"] == relative]
        if len(matches) != 1:
            raise ValueError(f"handoff adoption lacks one {relative}")
        return self.evidence.get(matches[0]["sha256"])

    def _adoption_entries(self, event: dict[str, Any]) -> dict[str, bytes]:
        adoption = self._verify_adoption_manifest(
            self.evidence.get(event["adoption_manifest_sha256"])
        )
        return {
            item["path"]: self.evidence.get(item["sha256"])
            for item in adoption["entries"]
        }

    def _verify_copied_campaign(
        self,
        event: dict[str, Any],
        genesis: dict[str, Any],
    ) -> None:
        entries = self._adoption_entries(event)
        manifest_bytes = entries.get("manifest.json")
        if manifest_bytes is None or sha256(manifest_bytes) != event[
            "campaign_bundle_manifest_sha256"
        ]:
            raise ValueError("copied campaign manifest differs from its adoption")
        manifest = json.loads(manifest_bytes)
        declared = {
            item["path"]: (item["sha256"], item["size_bytes"])
            for item in manifest["entries"]
        }
        copied = {
            name: (sha256(value), len(value))
            for name, value in entries.items()
            if name != "manifest.json"
        }
        protocol = StageDProtocolManifest.from_bytes(entries["protocol/manifest.json"])
        if (
            declared != copied
            or manifest["protocol_manifest_sha256"]
            != genesis["protocol_manifest_sha256"]
            or protocol.manifest_sha256 != genesis["protocol_manifest_sha256"]
        ):
            raise ValueError("copied campaign semantic closure differs")

    def _verify_copied_training(
        self,
        event: dict[str, Any],
        genesis: dict[str, Any],
        campaign: tuple[dict[str, Any], str] | None,
    ) -> None:
        if campaign is None:
            raise ValueError("copied training lacks its campaign")
        entries = self._adoption_entries(event)
        completion_bytes = entries.get("completion.json")
        if completion_bytes is None:
            raise ValueError("copied training lacks its completion")
        completion = StageDTrainingCompletion.from_bytes(completion_bytes)
        record_names = tuple(
            f"records/{index:08d}.json" for index in range(completion.trainer_record_count)
        )
        if (
            completion.completion_sha256 != event["training_completion_sha256"]
            or completion.trainer_ledger_head_sha256
            != event["trainer_ledger_head_sha256"]
            or completion.trainer_record_count != event["trainer_record_count"]
            or completion.protocol_manifest_sha256 != genesis["protocol_manifest_sha256"]
            or completion.campaign_manifest_sha256
            != campaign[0]["campaign_bundle_manifest_sha256"]
            or tuple(sha256(entries[name]) for name in record_names)
            != completion.record_sha256s
            or tuple(
                sorted(
                    name.removeprefix("evidence/")
                    for name in entries
                    if name.startswith("evidence/")
                )
            )
            != completion.evidence_sha256s
        ):
            raise ValueError("copied training semantic closure differs")

    def _verify_copied_evaluation(
        self,
        event: dict[str, Any],
        authorization: tuple[dict[str, Any], str] | None,
    ) -> None:
        if authorization is None:
            raise ValueError("copied evaluation lacks its authorization")
        entries = self._adoption_entries(event)
        completion_bytes = entries.get("completion.json")
        if completion_bytes is None:
            raise ValueError("copied evaluation lacks its completion")
        completion = StageDSealedEvaluationCompletion.from_bytes(completion_bytes)
        record_names = tuple(
            f"records/{index:08d}.json"
            for index in range(completion.evaluation_record_count)
        )
        if (
            sha256(completion_bytes) != event["evaluation_completion_sha256"]
            or completion.evaluation_ledger_head_sha256
            != event["evaluation_ledger_head_sha256"]
            or completion.evaluation_record_count != event["evaluation_record_count"]
            or completion.evaluation_authorization_sha256
            != authorization[0]["evaluation_authorization_sha256"]
            or tuple(sha256(entries[name]) for name in record_names)[-1]
            != completion.evaluation_ledger_head_sha256
        ):
            raise ValueError("copied evaluation semantic closure differs")

    @staticmethod
    def _required_digest(value: str | None) -> str:
        if value is None:
            raise RuntimeError("handoff state lacks a required digest")
        return value

    def adopt_evaluation(
        self,
        ledger: StageDEvaluationLedger,
        completion_bytes: bytes,
    ) -> str:
        completion = StageDSealedEvaluationCompletion.from_bytes(completion_bytes)
        completion.verify_ledger(ledger)
        snapshot = self.inspect()
        if (
            ledger.root.name != snapshot.evaluation_ledger_id
            or completion.execution_manifest_sha256 != snapshot.execution_manifest_sha256
            or completion.evaluation_authorization_sha256
            != snapshot.evaluation_authorization_sha256
        ):
            raise ValueError("evaluation completion differs from handoff authorization")
        allowed = {"inputs", "records", "responses", "evidence", "writer.lock"}
        if any(item.name not in allowed for item in ledger.root.iterdir()):
            raise ValueError("evaluation ledger has unknown top-level state")
        entries = [("completion.json", completion_bytes)]
        for directory_name in ("inputs", "records", "responses", "evidence"):
            directory = ledger.root / directory_name
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("evaluation bundle contains non-regular state")
                entries.append((f"{directory_name}/{path.name}", path.read_bytes()))
        adoption_sha256 = self._install_adoption("evaluation", entries)
        event = {
            "evaluation_completion_sha256": sha256(completion_bytes),
            "adoption_manifest_sha256": adoption_sha256,
            "evaluation_ledger_head_sha256": completion.evaluation_ledger_head_sha256,
            "evaluation_record_count": completion.evaluation_record_count,
        }
        return self._transition("evaluation_adopted", event, requires="evaluation_authorized")

    def materialize_evaluation_checkpoint(
        self,
        arm: ArmName,
        destination: Path,
    ) -> StageDCheckpointManifest:
        training = self.inspect().event("training_adopted")
        if training is None:
            raise RuntimeError("handoff lacks adopted training checkpoints")
        return materialize_adopted_checkpoint(
            training_entries=self._adoption_entries(training[0]),
            arm=arm,
            destination=destination,
        )

    def commit_report(
        self,
        *,
        report_bytes: bytes,
        decision_evidence_bytes: bytes,
        billing_evidence_bytes: bytes,
    ) -> str:
        del report_bytes, decision_evidence_bytes, billing_evidence_bytes
        raise RuntimeError("arbitrary report commits are disabled; use finalize_terminal")

    def finalize_terminal(
        self,
        *,
        terminal_status: TerminalStatus,
        terminal_phase: TerminalPhase,
        termination_code: TerminationCode,
        decisions: StageDDecisionVector,
        decision_evidence: Mapping[str, bytes],
        billing: StageDProviderBilling,
        billing_receipts: Mapping[str, bytes],
        cleanup: StageDCleanupEvidence,
        cleanup_receipts: Mapping[str, bytes],
        evaluation_ledger: StageDEvaluationLedger | None = None,
        evaluation_completion_bytes: bytes | None = None,
    ) -> bytes:
        return finalize_stage_d(
            cast(HandoffCoordinator, self),
            terminal_status=terminal_status,
            terminal_phase=terminal_phase,
            termination_code=termination_code,
            decisions=decisions,
            decision_evidence=decision_evidence,
            billing=billing,
            billing_receipts=billing_receipts,
            cleanup=cleanup,
            cleanup_receipts=cleanup_receipts,
            evaluation_ledger=evaluation_ledger,
            evaluation_completion_bytes=evaluation_completion_bytes,
        )

    def seal(self) -> bytes:
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            if snapshot.sealed:
                return (self.records / f"{snapshot.record_count - 1:08d}.json").read_bytes()
            report = snapshot.event("report_committed")
            if report is None:
                raise RuntimeError("handoff cannot seal before its report")
            self._append_unlocked("seal", {"report_record_sha256": report[1]})
            return (self.records / f"{snapshot.record_count:08d}.json").read_bytes()

    def _event_value(self, kind: str, name: str) -> Any:
        event = self.inspect().event(kind)
        if event is None:
            raise RuntimeError(f"handoff lacks required {kind} event")
        return event[0][name]

    def _transition(self, kind: str, event: dict[str, Any], *, requires: str) -> str:
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            existing = snapshot.event(kind)
            if existing is not None:
                if existing[0] != event:
                    raise FileExistsError(f"handoff {kind} differs from durable state")
                return existing[1]
            if snapshot.sealed or snapshot.events[-1][0] != requires:
                raise RuntimeError(f"handoff {kind} is out of order")
            return self._append_unlocked(kind, event)

    def _install_adoption(self, kind: str, entries: list[tuple[str, bytes]]) -> str:
        names = tuple(name for name, _ in entries)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            entries = sorted(entries)
            names = tuple(name for name, _ in entries)
        if len(names) != len(set(names)):
            raise ValueError("handoff adoption paths are duplicated")
        manifest_entries = []
        for relative, value in entries:
            _safe_relative(relative)
            digest = self._put_evidence(value)
            manifest_entries.append({"path": relative, "sha256": digest, "size_bytes": len(value)})
        manifest = canonical_json(
            {
                "schema_version": 1,
                "domain": _ADOPTION_DOMAIN,
                "kind": kind,
                "entries": manifest_entries,
            }
        )
        self._verify_adoption_manifest(manifest)
        return self._put_evidence(manifest)

    def _put_evidence(self, value: bytes) -> str:
        return self.evidence.put(value, fault_hook=self._fault_hook)

    def _verify_adoption_manifest(self, value: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("handoff adoption manifest is not JSON") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "domain", "kind", "entries"}
            or payload.get("schema_version") != 1
            or payload.get("domain") != _ADOPTION_DOMAIN
            or payload.get("kind") not in {"campaign", "training", "evaluation"}
            or not isinstance(payload.get("entries"), list)
            or canonical_json(payload) != value
        ):
            raise ValueError("handoff adoption manifest fields differ")
        names = []
        for item in payload["entries"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
                raise ValueError("handoff adoption entry fields differ")
            names.append(_safe_relative(item["path"]))
            evidence = self.evidence.get(item["sha256"])
            if type(item["size_bytes"]) is not int or item["size_bytes"] != len(evidence):
                raise ValueError("handoff adoption evidence size differs")
        if names != sorted(set(names)):
            raise ValueError("handoff adoption entry roster is not sorted and unique")
        return payload

    def _append_unlocked(self, kind: str, event: dict[str, Any]) -> str:
        paths = sorted(self.records.glob("*.json"))
        value = encode_record(
            offset=len(paths),
            prior_record_sha256=None if not paths else sha256(paths[-1].read_bytes()),
            record_kind=kind,
            event=event,
        )
        atomic_publish(
            self.records / f"{len(paths):08d}.json",
            value,
            fault_hook=self._fault_hook,
        )
        return sha256(value)


__all__ = [
    "AuthorizedEvaluation",
    "StageDHandoffCoordinator",
    "StageDHandoffSnapshot",
]
