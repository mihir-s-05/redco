"""Preparation and terminal verification for the bounded Stage-D E2 4B smoke."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_live_update import (
    LiveUpdateBinding,
    TrainerPoststep,
    TrainerPrestate,
    adapter_file_state_sha256,
)
from redco.analysis.stage_d_prime_bridge import audit_prime_cpu_batch, verify_prime_payload
from redco.analysis.stage_d_training_bridge import SealedTrainingBatch
from redco.analysis.stage_d_update_ledger import (
    SingleUseUpdateLedger,
    UpdateLedgerBinding,
)
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def base_snapshot_manifest(model_root: Path, *, revision: str) -> bytes:
    """Create a canonical file manifest for one exact locally downloaded snapshot."""
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("base model revision must be a full lowercase Git commit")
    files: list[dict[str, str | int]] = []
    for path in sorted(model_root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(model_root).parts:
            continue
        files.append(
            {
                "path": path.relative_to(model_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not files or not any(str(item["path"]).endswith(".safetensors") for item in files):
        raise ValueError("base snapshot manifest contains no model safetensors")
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-e2-base-snapshot-v1",
            "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": revision,
            "files": files,
        }
    )


def validate_prime_configs(trainer_path: Path, control_path: Path) -> dict[str, Any]:
    """Parse both frozen TOMLs through the pinned Prime Pydantic models."""
    trainer_module = importlib.import_module("prime_rl.configs.trainer")
    orchestrator_module = importlib.import_module("prime_rl.configs.orchestrator")
    trainer = trainer_module.TrainerConfig.model_validate(
        tomllib.loads(trainer_path.read_text(encoding="utf-8"))
    )
    control = orchestrator_module.OrchestratorConfig.model_validate(
        tomllib.loads(control_path.read_text(encoding="utf-8"))
    )
    expected_model = "/workspace/redco/.cache/models/qwen3-4b-instruct-2507-cdbee75f"
    if trainer.max_steps != 1 or control.max_steps != 1:
        raise ValueError("both Prime configs must permit exactly one optimizer step")
    if trainer.max_concurrent_runs != 1:
        raise ValueError("bounded Prime config must permit exactly one run")
    if str(trainer.model.name) != expected_model or str(control.model.name) != expected_model:
        raise ValueError("Prime configs differ from the frozen local model snapshot path")
    if control.output_dir != trainer.output_dir / "run_default":
        raise ValueError("Prime control output is not nested under the trainer output")
    if trainer.model.optim_cpu_offload or trainer.model.fsdp_cpu_offload:
        raise ValueError("bounded Prime config must keep model and optimizer state on GPU")
    return {
        "schema_version": 1,
        "analysis": "stage-d-e2-prime-config-parse-v1",
        "trainer_config_sha256": _file_sha256(trainer_path),
        "control_config_sha256": _file_sha256(control_path),
        "max_steps": trainer.max_steps,
        "max_concurrent_runs": trainer.max_concurrent_runs,
        "model": expected_model,
        "output_dir": str(trainer.output_dir),
    }


def prepare_live_inputs(
    *,
    sealed_batch_path: Path,
    golden_manifest_path: Path,
    trainer_config_path: Path,
    control_config_path: Path,
    base_manifest_path: Path,
    rollout_path: Path,
    binding_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    batch = SealedTrainingBatch.verify_bytes(sealed_batch_path.read_bytes())
    golden = _canonical_object(golden_manifest_path.read_bytes(), "golden manifest")
    if (
        golden.get("sealed_batch_sha256") != _sha256(sealed_batch_path.read_bytes())
        or golden.get("training_batch_identity") != batch.training_batch_identity
        or golden.get("bridge_payload_sha256") != batch.payload_sha256
    ):
        raise ValueError("golden manifest differs from the sealed training batch")
    runtime = golden.get("prime_runtime")
    if not isinstance(runtime, dict) or (
        _sha256(canonical_json(runtime)) != batch.binding.prime_runtime_sha256
    ):
        raise ValueError("golden runtime manifest differs from the sealed training batch")
    for category in ("sources", "patches"):
        values = runtime.get(category)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"golden runtime {category} manifest is missing")
        for path, expected in values.items():
            if not isinstance(path, str) or not isinstance(expected, str):
                raise ValueError(f"golden runtime {category} manifest is invalid")
            if _file_sha256(Path(path)) != expected:
                raise ValueError(f"live runtime file differs from the frozen manifest: {path}")
    if (
        golden.get("trainer_config_sha256") != _sha256(trainer_config_path.read_bytes())
        or golden.get("control_config_sha256") != _sha256(control_config_path.read_bytes())
    ):
        raise ValueError("live trainer or control config differs from the frozen golden manifest")
    prime = audit_prime_cpu_batch(batch)
    verify_prime_payload(prime, batch)
    base_manifest_bytes = base_manifest_path.read_bytes()
    base_manifest = _canonical_object(base_manifest_bytes, "base snapshot manifest")
    if base_manifest.get("revision") != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("base snapshot revision differs from the frozen Qwen revision")
    binding = LiveUpdateBinding(
        batch.binding.producer_seal_sha256,
        batch.training_batch_identity,
        batch.payload_sha256,
        prime.prime_payload_sha256,
        batch.binding.prime_runtime_sha256,
        batch.binding.trainer_config_sha256,
        _sha256(base_manifest_bytes),
        600,
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=False)
    rollout_path.write_bytes(prime.prime_payload)
    binding_path.write_bytes(binding.to_bytes())
    result = {
        "schema_version": 1,
        "analysis": "stage-d-e2-live-preflight-v1",
        "training_batch_identity": batch.training_batch_identity,
        "bridge_payload_sha256": batch.payload_sha256,
        "prime_payload_sha256": prime.prime_payload_sha256,
        "prime_sample_count": prime.sample_count,
        "prime_packed_record_count": prime.packed_record_count,
        "rl_normalizer_sum": prime.rl_normalizer_sum,
        "prime_normalized_loss": prime.prime_normalized_loss,
        "manual_normalized_loss": prime.manual_normalized_loss,
        "base_snapshot_manifest_sha256": _sha256(base_manifest_bytes),
        "live_binding_sha256": _sha256(binding.to_bytes()),
        "rollout_sha256": _sha256(prime.prime_payload),
        "trainer_config_sha256": _sha256(trainer_config_path.read_bytes()),
        "control_config_sha256": _sha256(control_config_path.read_bytes()),
    }
    preflight_path.write_bytes(canonical_json(result))
    return result


def verify_terminal_run(
    *,
    sealed_batch_path: Path,
    binding_path: Path,
    prestate_path: Path,
    poststep_path: Path,
    ledger_root: Path,
    metrics_path: Path,
    token_export_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    batch = SealedTrainingBatch.verify_bytes(sealed_batch_path.read_bytes())
    binding = LiveUpdateBinding.verify_bytes(binding_path.read_bytes())
    prestate = TrainerPrestate.verify_bytes(prestate_path.read_bytes())
    poststep = TrainerPoststep.verify_bytes(poststep_path.read_bytes())
    expected_ledger_binding = UpdateLedgerBinding(
        binding.producer_seal_sha256,
        binding.training_batch_identity,
        binding.bridge_payload_sha256,
        binding.prime_payload_sha256,
        binding.prime_runtime_sha256,
        binding.trainer_config_sha256,
        prestate.pre_model_sha256,
    )
    with SingleUseUpdateLedger(ledger_root) as ledger:
        if ledger.status != "complete" or ledger.binding != expected_ledger_binding:
            raise ValueError("local live update ledger differs from the supplied binding")
    completion = _canonical_object(
        (ledger_root / "records" / "00000002.json").read_bytes(),
        "ledger completion",
    )
    completion_body = completion.get("body")
    if not isinstance(completion_body, dict) or (
        completion_body.get("post_model_sha256") != poststep.post_model_sha256
        or completion_body.get("post_optimizer_sha256") != poststep.post_optimizer_sha256
        or completion_body.get("step_evidence_sha256") != _sha256(poststep_path.read_bytes())
        or completion_body.get("successful_optimizer_steps") != 1
    ):
        raise ValueError("ledger completion differs from the supplied trainer poststep")
    if binding.training_batch_identity != batch.training_batch_identity:
        raise ValueError("live binding differs from the sealed training batch")
    if adapter_file_state_sha256(
        adapter_path,
        base_snapshot_manifest_sha256=binding.base_snapshot_manifest_sha256,
    ) != poststep.post_model_sha256:
        raise ValueError("retained adapter differs from the authorized post-step LoRA state")
    metrics = _metrics(metrics_path)
    grad_norms = [
        float(record["optim/grad_norm"])
        for record in metrics
        if "optim/grad_norm" in record
    ]
    if len(grad_norms) != 1 or not math.isfinite(grad_norms[0]) or grad_norms[0] <= 0:
        raise ValueError("trainer metrics do not prove one finite nonzero gradient")
    if grad_norms[0] != poststep.gradient_l2:
        raise ValueError("trainer metric gradient differs from the authorized poststep")
    _verify_token_export(batch, token_export_path)
    result = {
        "schema_version": 1,
        "analysis": "stage-d-e2-4b-live-terminal-v1",
        "decision": "pass",
        "scientific_scope": "engineering_only_no_on_policy_or_learning_claim",
        "training_batch_identity": batch.training_batch_identity,
        "ledger_status": "complete",
        "optimizer_steps": [1],
        "gradient_l2": grad_norms[0],
        "prestate_sha256": _sha256(prestate_path.read_bytes()),
        "poststep_sha256": _sha256(poststep_path.read_bytes()),
        "post_model_sha256": poststep.post_model_sha256,
        "adapter_sha256": _sha256(adapter_path.read_bytes()),
        "token_export_sha256": _sha256(token_export_path.read_bytes()),
        "token_export_record_count": len(batch.records),
    }
    return result


def _metrics(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("trainer metric must be an object")
        records.append(value)
    if not records or {record.get("step") for record in records} != {1}:
        raise ValueError("trainer metrics must contain exactly optimizer step one")
    return records


def _verify_token_export(batch: SealedTrainingBatch, path: Path) -> None:
    if not (path.parent / "STABLE").is_file():
        raise ValueError("token export is missing its stable marker")
    exported = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("token export record must be an object")
        exported.append(value)
    if len(exported) != len(batch.records):
        raise ValueError("token export record count differs from the sealed bridge")
    by_tokens = {tuple(record.token_ids): record for record in batch.records}
    if len(by_tokens) != len(batch.records):
        raise ValueError("sealed E2 batch token streams must be unique")
    for value in exported:
        metadata = {
            "schema_version": 1,
            "step": 1,
            "export_step": 1,
            "rank": 0,
            "run_id": "run_default",
        }
        if any(value.get(name) != expected for name, expected in metadata.items()):
            raise ValueError("token export metadata differs from the frozen single-run contract")
        token_ids = tuple(value.get("token_ids", []))
        record = by_tokens.pop(token_ids, None)
        if record is None:
            raise ValueError("token export contains an unknown token stream")
        exact = {
            "loss_mask": list(record.mask),
            "env_name": record.env_name,
            "rl_normalizer": float(record.rl_normalizer),
        }
        if any(value.get(name) != expected for name, expected in exact.items()):
            raise ValueError("token export changed an exact sealed training field")
        for name, expected in (
            ("inference_logprobs", record.behavior_logprobs),
            ("temperatures", record.temperatures),
            ("advantages", record.advantages),
            ("rl_weights", record.rl_weights),
        ):
            observed = value.get(name)
            if not isinstance(observed, list) or len(observed) != len(expected):
                raise ValueError(f"token export {name} lost alignment")
            if any(
                not math.isclose(float(actual), target, rel_tol=1e-6, abs_tol=1e-7)
                for actual, target in zip(observed, expected, strict=True)
            ):
                raise ValueError(f"token export changed sealed {name}")
    if by_tokens:
        raise ValueError("token export omitted sealed training records")


def _canonical_object(value: bytes, label: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"{label} must be canonical JSON")
    return parsed
