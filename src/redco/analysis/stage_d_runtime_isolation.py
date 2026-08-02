"""Frozen runtime and workspace identity for isolated Stage-D rollouts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from redco.contracts import canonical_json

_IMAGE_DIGEST_PREFIX = "@sha256:"


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


def _canonical_mapping(value: Mapping[str, Any], name: str) -> bytes:
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return canonical_json(dict(value))


@dataclass(frozen=True, slots=True)
class StageDIsolatedRuntimeContract:
    """Exact Docker policy required for every QA, branch, and evaluation rollout."""

    image: str
    workdir: str = "/workspace"
    execution_user: str = "65534:65534"
    execution_home: str = "/tmp/redco-agent"
    context_path: str = "/workspace/evidence_context.txt"
    scratch_path: str = "/workspace/.rlm"

    def __post_init__(self) -> None:
        image_name, marker, digest = self.image.partition(_IMAGE_DIGEST_PREFIX)
        if not image_name or not marker:
            raise ValueError("isolated runtime image must be pinned by SHA-256 digest")
        _require_sha256(digest, "runtime image digest")
        expected = {
            "workdir": "/workspace",
            "execution_user": "65534:65534",
            "execution_home": "/tmp/redco-agent",
            "context_path": "/workspace/evidence_context.txt",
            "scratch_path": "/workspace/.rlm",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"isolated runtime {name} differs from frozen policy")

    def verify_runtime_config(self, value: Mapping[str, Any]) -> bytes:
        """Reject any runtime that is shared, privileged, mutable-tagged, or network-open."""
        config = dict(value)
        if config.get("type") != "docker":
            raise ValueError("Stage-D science requires a fresh Docker runtime per rollout")
        if config.get("image") != self.image:
            raise ValueError("Docker image differs from the frozen digest")
        if config.get("workdir") != self.workdir:
            raise ValueError("Docker workdir differs from the isolated workspace")
        if config.get("allow") != [] or config.get("block") != []:
            raise ValueError("Docker base network policy must allow framework routes only")
        if config.get("execution_user") != self.execution_user:
            raise ValueError("Docker execution user differs from the frozen non-root user")
        if config.get("execution_home") != self.execution_home:
            raise ValueError("Docker execution home differs from the private scratch home")
        if config.get("gpu") is not None:
            raise ValueError("RLM harness runtime must not inherit a GPU device")
        return canonical_json(config)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "domain": "redco-stage-d-isolated-runtime-contract-v1",
            "runtime_type": "docker",
            "image": self.image,
            "workdir": self.workdir,
            "execution_user": self.execution_user,
            "execution_home": self.execution_home,
            "context": {
                "path": self.context_path,
                "owner": "0:0",
                "mode": "0444",
            },
            "scratch": {
                "path": self.scratch_path,
                "owner": self.execution_user,
                "mode": "0700",
            },
            "network": {
                "base_allow": [],
                "base_block": [],
                "execution_allow": "framework_routes_only",
            },
            "fresh_runtime_per_rollout": True,
            "borrowed_runtime_forbidden": True,
        }

    @property
    def digest(self) -> str:
        return _sha256(canonical_json(self.to_payload()))


def build_pre_action_runtime_snapshot(
    *,
    contract: StageDIsolatedRuntimeContract,
    runtime_config: Mapping[str, Any],
    task_data: Mapping[str, Any],
    task_config: Mapping[str, Any],
    paper: bytes,
    frozen_workspace_manifest: Mapping[str, Any],
) -> bytes:
    """Bind every outcome-independent runtime/input byte into the action snapshot."""
    if type(paper) is not bytes or not paper:
        raise ValueError("runtime snapshot paper must be nonempty immutable bytes")
    runtime_config_bytes = contract.verify_runtime_config(runtime_config)
    task_data_bytes = _canonical_mapping(task_data, "task data")
    task_config_bytes = _canonical_mapping(task_config, "task config")
    workspace_bytes = _canonical_mapping(
        frozen_workspace_manifest,
        "frozen workspace manifest",
    )
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-pre-action-runtime-snapshot-v1",
            "contract": contract.to_payload(),
            "contract_sha256": contract.digest,
            "runtime_config_sha256": _sha256(runtime_config_bytes),
            "task_data_sha256": _sha256(task_data_bytes),
            "task_config_sha256": _sha256(task_config_bytes),
            "paper": {
                "path": contract.context_path,
                "sha256": _sha256(paper),
                "size_bytes": len(paper),
                "owner": "0:0",
                "mode": "0444",
            },
            "frozen_workspace_manifest_sha256": _sha256(workspace_bytes),
            "scratch": {
                "path": contract.scratch_path,
                "owner": contract.execution_user,
                "mode": "0700",
                "restored_fresh": True,
            },
            "network": {
                "policy": "framework_routes_only",
                "agent_egress": False,
            },
        }
    )


class _RuntimeResult(Protocol):
    exit_code: int
    stdout: str
    stderr: str


class IsolatedRuntimeLike(Protocol):
    async def run(self, argv: list[str], env: dict[str, str]) -> _RuntimeResult: ...


async def run_isolated_runtime_preflight(runtime: IsolatedRuntimeLike) -> bytes:
    """Prove the post-cut process boundary before the first model call."""
    script = r'''set -eu
test "$(id -u)" = 65534
test "$(id -g)" = 65534
test "$HOME" = /tmp/redco-agent
test "$XDG_CACHE_HOME" = /tmp/redco-agent/.cache
test "$(pwd)" = /workspace
test "$(stat -c '%u:%g:%a' /workspace/evidence_context.txt)" = 0:0:444
test -r /workspace/evidence_context.txt
test ! -w /workspace/evidence_context.txt
probe=/workspace/.rlm/.redco-preflight
printf redco-preflight > "$probe"
test "$(cat "$probe")" = redco-preflight
rm -f "$probe"
python - <<'PY'
import socket
try:
    connection = socket.create_connection(("1.1.1.1", 80), timeout=0.5)
except OSError:
    pass
else:
    connection.close()
    raise SystemExit("direct external socket unexpectedly succeeded")
PY
printf 'uid=65534\ngid=65534\nhome=/tmp/redco-agent\nnetwork=direct-blocked\n'
'''
    result = await runtime.run(["sh", "-c", script], {})
    if result.exit_code != 0:
        raise RuntimeError(
            "isolated runtime preflight failed: "
            + result.stderr[-1000:]
        )
    expected = (
        "uid=65534\n"
        "gid=65534\n"
        "home=/tmp/redco-agent\n"
        "network=direct-blocked\n"
    )
    if result.stdout != expected:
        raise RuntimeError("isolated runtime preflight returned unexpected evidence")
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-isolated-runtime-preflight-v1",
            "execution_user": "65534:65534",
            "execution_home": "/tmp/redco-agent",
            "context_owner_mode": "0:0:0444",
            "scratch_roundtrip": True,
            "direct_external_socket_blocked": True,
            "stdout_sha256": _sha256(result.stdout.encode("utf-8")),
        }
    )
