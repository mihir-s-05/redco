"""Canonical contracts for the single-use Stage-D held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast
from urllib.parse import urlsplit

from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json

_MANIFEST_DOMAIN = "redco-stage-d-evaluation-execution-manifest-v5"
_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")
_ROLES = {"client", "server"}
_ALLOWED_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
    "VLLM_NO_USAGE_STATS",
    "XDG_CACHE_HOME",
}
_SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "AUTH")
_RUNTIME_ENTRYPOINT_ROLES = ("task_runner", "scorer", "request_serializer")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())
    ):
        raise ValueError(f"{name} must be an absolute path")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeEntrypoint:
    role: Literal["task_runner", "scorer", "request_serializer"]
    member_path: str
    module: str
    callable_name: str
    api_schema: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.role not in _RUNTIME_ENTRYPOINT_ROLES:
            raise ValueError("evaluation runtime entrypoint role is invalid")
        path = PurePosixPath(self.member_path)
        expected_path = f"{self.module.replace('.', '/')}.py"
        if (
            not self.member_path
            or path.is_absolute()
            or "\\" in self.member_path
            or path.as_posix() != self.member_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(not part.isidentifier() for part in self.module.split("."))
            or self.member_path != expected_path
        ):
            raise ValueError("evaluation runtime entrypoint path differs from its module")
        if not self.callable_name.isidentifier() or not self.api_schema.isprintable():
            raise ValueError("evaluation runtime entrypoint callable or schema is invalid")
        _require_sha256(self.source_sha256, "evaluation runtime entrypoint source")

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "member_path": self.member_path,
            "module": self.module,
            "callable_name": self.callable_name,
            "api_schema": self.api_schema,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationScheduleUnit:
    ordinal: int
    arm: ArmName
    task_index: int
    task_id: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("evaluation schedule ordinal is invalid")
        if self.arm not in _ARMS:
            raise ValueError("evaluation schedule arm is invalid")
        if type(self.task_index) is not int or self.task_index < 0:
            raise ValueError("evaluation task index is invalid")
        if not self.task_id or not self.task_id.isprintable():
            raise ValueError("evaluation task ID is invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("evaluation task seed is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "arm": self.arm,
            "task_index": self.task_index,
            "task_id": self.task_id,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, value: object) -> EvaluationScheduleUnit:
        if not isinstance(value, dict) or set(value) != {
            "ordinal",
            "arm",
            "task_index",
            "task_id",
            "seed",
        }:
            raise ValueError("evaluation schedule unit fields differ")
        return cls(
            value["ordinal"],
            cast(ArmName, value["arm"]),
            value["task_index"],
            value["task_id"],
            value["seed"],
        )


@dataclass(frozen=True, slots=True)
class EvaluationProgramBinding:
    arm: ArmName
    role: Literal["client", "server"]
    absolute_executable: str
    executable_sha256: str
    argv: tuple[str, ...]
    working_directory: str
    checkpoint_root: str
    environment: tuple[tuple[str, str], ...]
    source_sha256s: tuple[tuple[str, str], ...]
    checkpoint_manifest_sha256: str
    post_model_sha256: str
    reload_evidence_sha256: str
    endpoint: str
    gpu_assignment: tuple[int, ...]
    cache_namespace: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS or self.role not in _ROLES:
            raise ValueError("evaluation program identity is invalid")
        _absolute(self.absolute_executable, "evaluation executable")
        _absolute(self.working_directory, "evaluation working directory")
        _absolute(self.checkpoint_root, "evaluation checkpoint root")
        _require_sha256(self.executable_sha256, "evaluation executable sha256")
        if (
            not self.argv
            or self.argv[0] != self.absolute_executable
            or any(not isinstance(item, str) or not item.isprintable() for item in self.argv)
        ):
            raise ValueError("evaluation program argv is invalid")
        if self.role == "server" and self.argv.count(self.checkpoint_root) != 1:
            raise ValueError("evaluation server argv must bind its checkpoint exactly once")
        names = tuple(name for name, _ in self.environment)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("evaluation environment names must be sorted and unique")
        for name, value in self.environment:
            if (
                name not in _ALLOWED_ENVIRONMENT
                or any(fragment in name.upper() for fragment in _SECRET_FRAGMENTS)
                or not isinstance(value, str)
                or "\x00" in value
            ):
                raise ValueError(f"evaluation environment entry is forbidden: {name}")
        source_names = tuple(name for name, _ in self.source_sha256s)
        if (
            not source_names
            or source_names != tuple(sorted(source_names))
            or len(source_names) != len(set(source_names))
        ):
            raise ValueError("evaluation program sources must be sorted and unique")
        for name, digest in self.source_sha256s:
            posix_name = PurePosixPath(name)
            windows_name = PureWindowsPath(name)
            if (
                not name
                or not name.isprintable()
                or posix_name.is_absolute()
                or windows_name.is_absolute()
                or ".." in posix_name.parts
                or ".." in windows_name.parts
            ):
                raise ValueError("evaluation source name is invalid")
            _require_sha256(digest, f"evaluation source {name}")
        for name in (
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "reload_evidence_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        try:
            endpoint = urlsplit(self.endpoint)
            port = endpoint.port
        except ValueError as error:
            raise ValueError("evaluation endpoint must be loopback HTTP") from error
        if (
            endpoint.scheme != "http"
            or endpoint.hostname != "127.0.0.1"
            or port is None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("evaluation endpoint must be loopback HTTP")
        if (
            not self.gpu_assignment
            or len(set(self.gpu_assignment)) != len(self.gpu_assignment)
            or any(type(item) is not int or item < 0 for item in self.gpu_assignment)
        ):
            raise ValueError("evaluation GPU assignment is invalid")
        if not self.cache_namespace or not self.cache_namespace.isprintable():
            raise ValueError("evaluation cache namespace is invalid")

    @property
    def binding_sha256(self) -> str:
        return _sha256(canonical_json(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "role": self.role,
            "absolute_executable": self.absolute_executable,
            "executable_sha256": self.executable_sha256,
            "argv": list(self.argv),
            "working_directory": self.working_directory,
            "checkpoint_root": self.checkpoint_root,
            "environment": dict(self.environment),
            "source_sha256s": dict(self.source_sha256s),
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "post_model_sha256": self.post_model_sha256,
            "reload_evidence_sha256": self.reload_evidence_sha256,
            "endpoint": self.endpoint,
            "gpu_assignment": list(self.gpu_assignment),
            "cache_namespace": self.cache_namespace,
        }

    @classmethod
    def from_payload(cls, value: object) -> EvaluationProgramBinding:
        fields = {
            "arm",
            "role",
            "absolute_executable",
            "executable_sha256",
            "argv",
            "working_directory",
            "checkpoint_root",
            "environment",
            "source_sha256s",
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "reload_evidence_sha256",
            "endpoint",
            "gpu_assignment",
            "cache_namespace",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or not isinstance(value.get("argv"), list)
            or not isinstance(value.get("environment"), dict)
            or not isinstance(value.get("source_sha256s"), dict)
            or not isinstance(value.get("gpu_assignment"), list)
        ):
            raise ValueError("evaluation program binding fields differ")
        environment = value["environment"]
        sources = value["source_sha256s"]
        return cls(
            arm=cast(ArmName, value["arm"]),
            role=cast(Literal["client", "server"], value["role"]),
            absolute_executable=value["absolute_executable"],
            executable_sha256=value["executable_sha256"],
            argv=tuple(value["argv"]),
            working_directory=value["working_directory"],
            checkpoint_root=value["checkpoint_root"],
            environment=tuple(sorted(environment.items())),
            source_sha256s=tuple(sorted(sources.items())),
            checkpoint_manifest_sha256=value["checkpoint_manifest_sha256"],
            post_model_sha256=value["post_model_sha256"],
            reload_evidence_sha256=value["reload_evidence_sha256"],
            endpoint=value["endpoint"],
            gpu_assignment=tuple(value["gpu_assignment"]),
            cache_namespace=value["cache_namespace"],
        )


@dataclass(frozen=True, slots=True)
class EvaluationSupervisorLimits:
    control_root: str
    log_root: str
    cgroup_root: str
    evaluation_timeout_seconds: int
    claim_timeout_seconds: int
    probe_timeout_seconds: int
    stop_timeout_seconds: int
    poll_interval_milliseconds: int
    max_log_bytes: int

    def __post_init__(self) -> None:
        _absolute(self.control_root, "evaluation control root")
        _absolute(self.log_root, "evaluation log root")
        _absolute(self.cgroup_root, "evaluation cgroup root")
        for name in (
            "evaluation_timeout_seconds",
            "claim_timeout_seconds",
            "probe_timeout_seconds",
            "stop_timeout_seconds",
            "poll_interval_milliseconds",
            "max_log_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.poll_interval_milliseconds > 1000:
            raise ValueError("evaluation supervisor polling may not exceed one second")

    def to_payload(self) -> dict[str, object]:
        return {
            "control_root": self.control_root,
            "log_root": self.log_root,
            "cgroup_root": self.cgroup_root,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "claim_timeout_seconds": self.claim_timeout_seconds,
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "stop_timeout_seconds": self.stop_timeout_seconds,
            "poll_interval_milliseconds": self.poll_interval_milliseconds,
            "max_log_bytes": self.max_log_bytes,
        }

    @classmethod
    def from_payload(cls, value: object) -> EvaluationSupervisorLimits:
        fields = {
            "control_root",
            "log_root",
            "cgroup_root",
            "evaluation_timeout_seconds",
            "claim_timeout_seconds",
            "probe_timeout_seconds",
            "stop_timeout_seconds",
            "poll_interval_milliseconds",
            "max_log_bytes",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("evaluation supervisor limits fields differ")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StageDEvaluationExecutionManifest:
    evaluation_ledger_id: str
    protocol_manifest_sha256: str
    trainer_ledger_head_sha256: str
    trainer_record_count: int
    heldout_eval_config_sha256: str
    evaluation_plan_sha256: str
    decision_rule_sha256: str
    runtime_entrypoints: tuple[EvaluationRuntimeEntrypoint, ...]
    runtime_worker_image: str
    runtime_bundle_path: str
    runtime_bundle_sha256: str
    container_runtime_executable: str
    container_runtime_executable_sha256: str
    supervisor_limits: EvaluationSupervisorLimits
    max_server_launches_per_arm: int
    max_client_launches_per_arm: int
    server_replacement_policy: str
    programs: tuple[EvaluationProgramBinding, ...]
    schedule: tuple[EvaluationScheduleUnit, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.evaluation_ledger_id, "evaluation ledger ID")
        for name in (
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
            "heldout_eval_config_sha256",
            "evaluation_plan_sha256",
            "decision_rule_sha256",
            "runtime_bundle_sha256",
            "container_runtime_executable_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.trainer_record_count) is not int or self.trainer_record_count < 1:
            raise ValueError("evaluation trainer record count is invalid")
        _absolute(self.runtime_bundle_path, "evaluation runtime bundle path")
        _absolute(
            self.container_runtime_executable,
            "evaluation container runtime executable",
        )
        image_parts = self.runtime_worker_image.rsplit("@sha256:", 1)
        if len(image_parts) != 2 or not image_parts[0]:
            raise ValueError("evaluation runtime worker image is not digest-pinned")
        _require_sha256(image_parts[1], "evaluation runtime worker image digest")
        if (
            type(self.max_server_launches_per_arm) is not int
            or self.max_server_launches_per_arm not in {1, 2}
            or self.server_replacement_policy != "before-first-dispatch-only-v1"
        ):
            raise ValueError("evaluation server replacement policy is invalid")
        if type(
            self.max_client_launches_per_arm
        ) is not int or self.max_client_launches_per_arm not in {1, 2}:
            raise ValueError("evaluation client launch budget is invalid")
        if tuple(item.role for item in self.runtime_entrypoints) != _RUNTIME_ENTRYPOINT_ROLES:
            raise ValueError("evaluation runtime entrypoints differ from the exact role order")
        program_keys = tuple((item.arm, item.role) for item in self.programs)
        expected_program_keys = tuple((arm, role) for arm in _ARMS for role in ("server", "client"))
        if program_keys != expected_program_keys:
            raise ValueError("evaluation programs do not use the exact arm/role order")
        if not self.schedule or tuple(item.ordinal for item in self.schedule) != tuple(
            range(len(self.schedule))
        ):
            raise ValueError("evaluation schedule ordinals are not contiguous")
        per_arm: dict[ArmName, list[tuple[int, str, int]]] = {arm: [] for arm in _ARMS}
        for item in self.schedule:
            per_arm[item.arm].append((item.task_index, item.task_id, item.seed))
        reference: tuple[tuple[int, str, int], ...] | None = None
        for arm in _ARMS:
            values = tuple(per_arm[arm])
            if tuple(index for index, _, _ in values) != tuple(range(len(values))):
                raise ValueError("evaluation per-arm task indexes are not contiguous")
            if reference is None:
                reference = values
            elif values != reference:
                raise ValueError("evaluation arms do not share an exact task roster")
        assert reference is not None
        expected_arm_blocks = tuple(arm for arm in _ARMS for _ in range(len(reference)))
        if tuple(item.arm for item in self.schedule) != expected_arm_blocks:
            raise ValueError("evaluation schedule must use contiguous frozen arm blocks")
        program_by_key = {(item.arm, item.role): item for item in self.programs}
        for program in self.programs:
            environment = dict(program.environment)
            if environment.get("PYTHONPATH") not in (None, self.runtime_bundle_path):
                raise ValueError("evaluation PYTHONPATH is not the frozen runtime bundle")
        server_endpoints = tuple(program_by_key[(arm, "server")].endpoint for arm in _ARMS)
        cache_namespaces = tuple(program_by_key[(arm, "server")].cache_namespace for arm in _ARMS)
        if len(set(server_endpoints)) != len(server_endpoints) or len(set(cache_namespaces)) != len(
            cache_namespaces
        ):
            raise ValueError("evaluation arms must use isolated endpoints and cache namespaces")
        for arm in _ARMS:
            client = program_by_key[(arm, "client")]
            server = program_by_key[(arm, "server")]
            if (
                client.checkpoint_manifest_sha256 != server.checkpoint_manifest_sha256
                or client.post_model_sha256 != server.post_model_sha256
                or client.reload_evidence_sha256 != server.reload_evidence_sha256
                or client.checkpoint_root != server.checkpoint_root
                or client.endpoint != server.endpoint
                or client.gpu_assignment != server.gpu_assignment
                or client.cache_namespace != server.cache_namespace
            ):
                raise ValueError("evaluation client/server policy bindings differ")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 9,
                "domain": _MANIFEST_DOMAIN,
                "evaluation_ledger_id": self.evaluation_ledger_id,
                "protocol_manifest_sha256": self.protocol_manifest_sha256,
                "trainer_ledger_head_sha256": self.trainer_ledger_head_sha256,
                "trainer_record_count": self.trainer_record_count,
                "heldout_eval_config_sha256": self.heldout_eval_config_sha256,
                "evaluation_plan_sha256": self.evaluation_plan_sha256,
                "decision_rule_sha256": self.decision_rule_sha256,
                "runtime_entrypoints": [item.to_payload() for item in self.runtime_entrypoints],
                "runtime_worker_image": self.runtime_worker_image,
                "runtime_bundle_path": self.runtime_bundle_path,
                "runtime_bundle_sha256": self.runtime_bundle_sha256,
                "container_runtime_executable": self.container_runtime_executable,
                "container_runtime_executable_sha256": (self.container_runtime_executable_sha256),
                "supervisor_limits": self.supervisor_limits.to_payload(),
                "max_server_launches_per_arm": self.max_server_launches_per_arm,
                "max_client_launches_per_arm": self.max_client_launches_per_arm,
                "server_replacement_policy": self.server_replacement_policy,
                "programs": [item.to_payload() for item in self.programs],
                "schedule": [item.to_payload() for item in self.schedule],
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDEvaluationExecutionManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("evaluation execution manifest is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "evaluation_ledger_id",
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
            "trainer_record_count",
            "heldout_eval_config_sha256",
            "evaluation_plan_sha256",
            "decision_rule_sha256",
            "runtime_entrypoints",
            "runtime_worker_image",
            "runtime_bundle_path",
            "runtime_bundle_sha256",
            "container_runtime_executable",
            "container_runtime_executable_sha256",
            "supervisor_limits",
            "max_server_launches_per_arm",
            "max_client_launches_per_arm",
            "server_replacement_policy",
            "programs",
            "schedule",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 9
            or payload.get("domain") != _MANIFEST_DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("runtime_entrypoints"), list)
            or not isinstance(payload.get("programs"), list)
            or not isinstance(payload.get("schedule"), list)
        ):
            raise ValueError("evaluation execution manifest fields differ")
        return cls(
            evaluation_ledger_id=payload["evaluation_ledger_id"],
            protocol_manifest_sha256=payload["protocol_manifest_sha256"],
            trainer_ledger_head_sha256=payload["trainer_ledger_head_sha256"],
            trainer_record_count=payload["trainer_record_count"],
            heldout_eval_config_sha256=payload["heldout_eval_config_sha256"],
            evaluation_plan_sha256=payload["evaluation_plan_sha256"],
            decision_rule_sha256=payload["decision_rule_sha256"],
            runtime_entrypoints=tuple(
                EvaluationRuntimeEntrypoint(**item) for item in payload["runtime_entrypoints"]
            ),
            runtime_worker_image=payload["runtime_worker_image"],
            runtime_bundle_path=payload["runtime_bundle_path"],
            runtime_bundle_sha256=payload["runtime_bundle_sha256"],
            container_runtime_executable=payload["container_runtime_executable"],
            container_runtime_executable_sha256=payload["container_runtime_executable_sha256"],
            supervisor_limits=EvaluationSupervisorLimits.from_payload(payload["supervisor_limits"]),
            max_server_launches_per_arm=payload["max_server_launches_per_arm"],
            max_client_launches_per_arm=payload["max_client_launches_per_arm"],
            server_replacement_policy=payload["server_replacement_policy"],
            programs=tuple(
                EvaluationProgramBinding.from_payload(item) for item in payload["programs"]
            ),
            schedule=tuple(
                EvaluationScheduleUnit.from_payload(item) for item in payload["schedule"]
            ),
        )

    def program(self, arm: ArmName, role: Literal["client", "server"]) -> EvaluationProgramBinding:
        return next(item for item in self.programs if item.arm == arm and item.role == role)


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluation_environment_sha256(environment: tuple[tuple[str, str], ...]) -> str:
    names = tuple(name for name, _ in environment)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("evaluation environment must be sorted and unique")
    for name, value in environment:
        if (
            name not in _ALLOWED_ENVIRONMENT
            or any(fragment in name.upper() for fragment in _SECRET_FRAGMENTS)
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError(f"evaluation environment entry is forbidden: {name}")
    return _sha256(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-environment-v1",
                "environment": dict(environment),
            }
        )
    )


__all__ = [
    "EvaluationProgramBinding",
    "EvaluationRuntimeEntrypoint",
    "EvaluationScheduleUnit",
    "EvaluationSupervisorLimits",
    "StageDEvaluationExecutionManifest",
    "evaluation_environment_sha256",
    "hash_file",
]
