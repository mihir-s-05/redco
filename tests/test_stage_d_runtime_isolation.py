from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

import pytest

from redco.analysis.stage_d_runtime_isolation import (
    StageDIsolatedRuntimeContract,
    build_pre_action_runtime_snapshot,
    run_isolated_runtime_preflight,
)

IMAGE = "python@sha256:" + "a" * 64


def _runtime() -> dict[str, object]:
    return {
        "type": "docker",
        "image": IMAGE,
        "workdir": "/workspace",
        "allow": [],
        "block": [],
        "gpu": None,
        "execution_user": "65534:65534",
        "execution_home": "/tmp/redco-agent",
    }


def test_runtime_contract_requires_fresh_closed_non_root_docker() -> None:
    contract = StageDIsolatedRuntimeContract(IMAGE)
    assert contract.verify_runtime_config(_runtime())
    for field, bad in (
        ("type", "subprocess"),
        ("image", "python:3.11-slim"),
        ("allow", ["*"]),
        ("execution_user", "0:0"),
        ("gpu", "1"),
    ):
        value = _runtime()
        value[field] = bad
        with pytest.raises(ValueError):
            contract.verify_runtime_config(value)


def test_pre_action_snapshot_binds_runtime_task_paper_and_workspace() -> None:
    contract = StageDIsolatedRuntimeContract(IMAGE)
    paper = b"immutable paper"
    value = build_pre_action_runtime_snapshot(
        contract=contract,
        runtime_config=_runtime(),
        task_data={"example_id": "e-1", "question": "q"},
        task_config={"split": "science_train"},
        paper=paper,
        frozen_workspace_manifest={
            "entries": [
                {
                    "path": "/workspace/evidence_context.txt",
                    "sha256": hashlib.sha256(paper).hexdigest(),
                    "mode": "0444",
                }
            ]
        },
    )
    payload = json.loads(value)
    assert payload["paper"]["sha256"] == hashlib.sha256(paper).hexdigest()
    assert payload["network"] == {
        "agent_egress": False,
        "policy": "framework_routes_only",
    }
    assert payload["scratch"]["restored_fresh"] is True

    changed = json.loads(value)
    changed["paper"]["size_bytes"] += 1
    assert changed != payload


@dataclass(frozen=True)
class _Result:
    exit_code: int
    stdout: str
    stderr: str = ""


class _Runtime:
    def __init__(self, result: _Result) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, argv: list[str], env: dict[str, str]) -> _Result:
        self.calls.append((argv, env))
        return self.result


def test_post_cut_runtime_preflight_requires_exact_boundary_evidence() -> None:
    stdout = (
        "uid=65534\n"
        "gid=65534\n"
        "home=/tmp/redco-agent\n"
        "network=direct-blocked\n"
    )
    runtime = _Runtime(_Result(0, stdout))
    report = json.loads(asyncio.run(run_isolated_runtime_preflight(runtime)))

    assert report["direct_external_socket_blocked"] is True
    assert report["scratch_roundtrip"] is True
    assert runtime.calls[0][0][:2] == ["sh", "-c"]
    assert "socket.create_connection" in runtime.calls[0][0][2]
    assert runtime.calls[0][1] == {}


@pytest.mark.parametrize(
    "result",
    [
        _Result(1, "", "wrong uid"),
        _Result(0, "uid=0\n"),
    ],
)
def test_post_cut_runtime_preflight_rejects_failure_or_ambiguous_output(
    result: _Result,
) -> None:
    with pytest.raises(RuntimeError, match="preflight"):
        asyncio.run(run_isolated_runtime_preflight(_Runtime(result)))
