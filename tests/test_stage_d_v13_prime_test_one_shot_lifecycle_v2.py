"""Adversarial source-free tests for the Prime test-only one-shot lifecycle."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from test_stage_d_v13_prime_test_one_shot_evidence_v2 import (
    _bind_journal,
    _Client,
    _Commands,
    _completed_with_history,
    _context,
    _failed_fixture,
    _item,
    _journal_records,
    _parametrize,
    _replace_bound_bytes,
    _replace_bound_json,
    _resign_terminal,
    _Response,
    _run_fixture,
    _Transport,
    _wallet,
    _wallet_snapshot,
)

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_evidence_v2 as evidence
from redco.analysis import stage_d_v13_prime_test_one_shot_handoff_v2 as handoff_owner
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis import stage_d_v13_prime_test_one_shot_prime_v2 as prime_owner
from redco.analysis import stage_d_v13_prime_test_one_shot_remote_v2 as remote_owner
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ASSESSMENT_TTL_SECONDS,
    CLEANUP_TIMEOUT_SECONDS,
    COMMAND_TIMEOUT_SECONDS,
    EVIDENCE_ROOT,
    HANDOFF_NAMESPACE,
    MAX_CLEANUP_PRIME_CLI_CALLS,
    PODS_CREATE_ENDPOINT,
    READINESS_AUTHORITY,
    RUNTIME_AUTHORITY,
    TERMINAL_NAMESPACE,
    canonical_json,
    exclusive_runtime_root,
    fixed_runtime_path,
    publish_once,
    sha256_bytes,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_binding_v2 import _RuntimeContext


@_parametrize(
    "value",
    [
        "-oProxyCommand=bad@8.8.8.8 -p 22",
        "-F@8.8.8.8 -p 22",
        "-f@8.8.8.8 -p 22",
        "ubuntu@8.8.8.8 -p 22,root@1.1.1.1 -p 22",
        "ubuntu@[2001:4860:::8888] -p 22",
        "ubuntu@127.0.0.1 -p 22",
        "ubuntu@203.0.113.10 -p 22",
        "ubuntu@169.254.1.1 -p 22",
    ],
)
def test_ssh_endpoint_grammar_rejects_options_aliases_and_nonpublic(value: str) -> None:
    with pytest.raises(RuntimeError):
        remote_owner.parse_endpoint({"sshConnection": value})
def test_ssh_endpoint_grammar_accepts_exact_public_ipv4_and_ipv6() -> None:
    assert remote_owner.parse_endpoint({"ssh": "ubuntu@8.8.8.8 -p 2222"}) == (
        "ubuntu",
        "8.8.8.8",
        2222,
    )
_CREATE_MUTATIONS = [
    ("create-dispatch", ("unknown",), None, False),
    ("create-dispatch", ("endpoint",), None, True),
    ("create-dispatch", ("schema_version",), True, False),
    ("create-dispatch", ("state",), "forged", False),
    ("create-dispatch", ("payload_sha256",), "0" * 64, False),
    ("create-dispatch", ("resource_sha256",), "1" * 64, False),
    ("create-dispatch", ("disk_size",), False, False),
    ("create-dispatch", ("attempt_limit",), True, False),
    ("create-dispatch", ("retry",), True, False),
    ("create-result", ("unknown",), None, False),
    ("create-result", ("status_code",), None, True),
    ("create-result", ("schema_version",), True, False),
    ("create-result", ("state",), "forged", False),
    ("create-result", ("status_code",), 202, False),
    ("create-result", ("response_bytes",), False, False),
    ("create-result", ("response_sha256",), "0" * 64, False),
    ("create-result", ("pod_identity_sha256",), "1" * 64, False),
    ("create-result", ("pod_name",), "wrong", False),
    *[
        (artifact, ("authority", name), True, False)
        for artifact in ("create-dispatch", "create-result")
        for name in READINESS_AUTHORITY
    ],
]
@_parametrize("artifact,path,replacement,delete", _CREATE_MUTATIONS)
def test_resigned_terminal_rejects_create_schema_and_binding_matrix(
    tmp_path: Path, artifact: str, path: tuple[str, ...], replacement: object, delete: bool
) -> None:
    context, root = _completed_with_history(tmp_path / f"{artifact}-{path[-1]}")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    value = cast(
        dict[str, Any],
        json.loads((root / evidence.ARTIFACT_FILENAMES[artifact]).read_bytes()),
    )
    target = value
    for key in path[:-1]:
        target = cast(dict[str, Any], target[key])
    target.pop(path[-1]) if delete else target.__setitem__(path[-1], replacement)
    _replace_bound_json(root, terminal, artifact, value)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
def test_create_dispatch_requires_authenticated_wallet_before(tmp_path: Path) -> None:
    context, root = _failed_fixture(tmp_path / "missing-before")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    (root / "wallet-before.json").unlink()
    cast(dict[str, Any], terminal["evidence_dag"]).pop("wallet-before")
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
def test_precreate_capacity_state_rejects_orphan_wallet_before(tmp_path: Path) -> None:
    _source_context, source_root = _failed_fixture(tmp_path / "valid-wallet-source")
    raw = (source_root / "wallet-before.json").read_bytes()
    repository = tmp_path / "orphan-before"
    repository.mkdir()
    context = _context(repository, _Transport([]), _Commands(repository))
    assert _run_fixture(context).state == "no_qualifying_capacity"
    root = repository / EVIDENCE_ROOT
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    (root / "wallet-before.json").write_bytes(raw)
    cast(dict[str, Any], terminal["evidence_dag"])["wallet-before"] = {
        "path": "wallet-before.json", "bytes": len(raw), "sha256": sha256_bytes(raw)
    }
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
def test_failed_cleanup_accepts_valid_wallet_before_without_after(tmp_path: Path) -> None:
    context, root = _failed_fixture(tmp_path / "valid-before")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    cleanup_value = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
    cleanup_value["wallet_after"] = None
    cleanup_value["errors"] = ["wallet:RuntimeError"]
    terminal["cleanup_proven"] = False
    terminal["cleanup_failures"] = ["wallet:RuntimeError"]
    _replace_bound_json(root, terminal, "cleanup", cleanup_value)
    _resign_terminal(context, root, terminal)
    assert evidence.verify_terminal_evidence(root, context.identity)["state"] == "failed_terminal"
    assert remote_owner.parse_endpoint({"ssh": "ubuntu@[2001:4860:4860::8888] -p 22"}) == (
        "ubuntu",
        "2001:4860:4860::8888",
        22,
    )
def test_multiple_distinct_capacity_selects_deterministically_but_duplicates_fail() -> None:
    transport = _Transport(
        [
            _item(1),
            _item(
                2,
                prices={
                    "onDemand": 1.9,
                    "communityPrice": 1.7,
                    "isVariable": False,
                    "currency": "USD",
                },
            ),
        ]
    )
    pages, diagnostic, failure, count = v5._capture_pages(cast(Any, _Client(transport)), (OSError,))
    assert (diagnostic, failure, count) == (None, None, 2)
    raw, resource = prime_owner.assess_pages(pages, {"commit": "a"}, 100)
    assert resource is not None and resource["cloudId"] == "cloud-1"
    assert json.loads(raw)["eligible_count"] == 2
    duplicate = _Transport([_item(1), _item(2, cloudId="cloud-1", provider="other-provider")])
    pages, _, _, _ = v5._capture_pages(cast(Any, _Client(duplicate)), (OSError,))
    raw, resource = prime_owner.assess_pages(pages, {"commit": "a"}, 100)
    assert resource is None and json.loads(raw)["state"] == "ambiguous_capacity"
def test_no_capacity_performs_no_prime_or_create_calls(tmp_path: Path) -> None:
    transport = _Transport([])
    commands = _Commands(tmp_path)
    result = _run_fixture(_context(tmp_path, transport, commands))
    assert result.state == "no_qualifying_capacity"
    assert len(transport.calls) == 2
    assert all(Path(call[0]).name.lower() == "ssh-keygen.exe" for call in commands.invocations)
    assert not result.create_dispatched
def test_zero_disk_direct_create_uses_assessed_resource_without_second_availability(
    tmp_path: Path,
) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    context = _context(tmp_path, transport, commands)
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    resource = _item(1)
    prime_owner.direct_create(
        owner,
        resource,
        "team-fixture",
        fixed_runtime_path(root, "create-dispatch.json"),
        fixed_runtime_path(root, "create-result.json"),
    )
    assert [call[0] for call in transport.calls] == ["POST"]
    assert transport.calls[0][1] == PODS_CREATE_ENDPOINT
    assert owner.create_dispatched
def test_expired_operational_deadline_blocks_create_before_dispatch(
    tmp_path: Path,
) -> None:
    transport = _Transport([_item(1)])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    object.__setattr__(context, "monotonic", lambda: 21_601.0)
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 0.0)
    dispatch = fixed_runtime_path(root, "create-dispatch.json")
    with pytest.raises(TimeoutError, match="before create"):
        prime_owner.direct_create(
            owner,
            _item(1),
            "team-fixture",
            dispatch,
            fixed_runtime_path(root, "create-result.json"),
        )
    assert not transport.calls and not dispatch.exists() and not owner.create_dispatched
@_parametrize(
    ("field", "replacement"),
    [
        ("name", "wrong"),
        ("gpuName", "A100 80GB"),
        ("gpuCount", 1),
        ("providerType", "wrong"),
        ("status", "UNKNOWN"),
        ("createdAt", ""),
    ],
)
def test_create_response_is_bound_to_selected_request(
    tmp_path: Path, field: str, replacement: object
) -> None:
    class MutatedTransport(_Transport):
        def request(self, method: str, url: str, **kwargs: object) -> _Response:
            response = super().request(method, url, **kwargs)
            if method != "POST":
                return response
            value = json.loads(response.content)
            value[field] = replacement
            return _Response(response.status_code, canonical_json(value), response.headers)
    transport = MutatedTransport([_item(1)])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    with pytest.raises(RuntimeError, match="schema"):
        prime_owner.direct_create(
            owner,
            _item(1),
            "team-fixture",
            fixed_runtime_path(root, "create-dispatch.json"),
            fixed_runtime_path(root, "create-result.json"),
        )
    assert owner.create_dispatched
def test_authenticated_create_id_must_equal_reconciled_exact_name_id(tmp_path: Path) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    context = _context(tmp_path, transport, commands)
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    owner.trusted_pod_id = "response-id"
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        args = tuple(argv)
        raw = canonical_json(
            {
                "pods": [{"id": "different-id", "name": owner.pod_name}],
                "total_count": 1,
                "offset": 0,
                "limit": 100,
            }
        )
        return lifecycle.CommandResult(args, 0, raw, b"")
    object.__setattr__(context, "run", run)
    with pytest.raises(RuntimeError, match="identity differ"):
        remote_owner.reconcile_created_pod(owner)
    assert owner.known_pod_ids == {"different-id"}
def test_ambiguous_create_is_dispatched_once_reconciled_and_terminated(
    tmp_path: Path,
) -> None:
    transport = _Transport([_item(1)], ambiguous_create=True)
    commands = _Commands(tmp_path)
    result = _run_fixture(_context(tmp_path, transport, commands))
    assert result.state == "failed_terminal"
    assert result.create_dispatched and result.cleanup_proven
    assert [method for method, _url, _payload in transport.calls].count("POST") == 1
    assert commands.terminated
    assert sum("terminate" in call for call in commands.invocations) == 1
def test_cleanup_adopts_late_exact_name_pods_once_and_ignores_unrelated(
    tmp_path: Path,
) -> None:
    transport = _Transport([])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000
    transport.wallet_handler = lambda _offset, _ordinal: _wallet(
        29.5, billed=True, resource_id="late-a"
    )
    polls = 0
    terminated: list[str] = []
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        nonlocal polls
        args = tuple(argv)
        rows: list[dict[str, object]] = []
        if "pods" in args and "list" in args:
            polls += 1
            if polls == 2:
                rows = [
                    {"id": "late-a", "name": owner.pod_name},
                    {"id": "late-b", "name": owner.pod_name},
                ]
            output = canonical_json(
                {"pods": rows, "total_count": len(rows), "offset": 0, "limit": 100}
            )
        elif "terminate" in args:
            terminated.append(args[args.index("terminate") + 1])
            output = b""
        elif "disks" in args:
            output = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        elif "wallet" in args:
            output = _wallet(29.5, billed=True)
        else:
            output = b""
        return lifecycle.CommandResult(args, 0, output, b"")
    object.__setattr__(context, "run", run)
    ok, evidence, errors = prime_owner.cleanup(owner, _wallet_snapshot(_wallet(30.0, billed=False)))
    assert ok and not errors and sorted(terminated) == ["late-a", "late-b"]
    assert len(cast(list[str], evidence["terminated_identity_sha256s"])) == 2
def test_cleanup_never_terminates_unrelated_pod_and_fails_global_zero(tmp_path: Path) -> None:
    transport = _Transport([])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000
    transport.wallet_handler = lambda _offset, _ordinal: _wallet(29.5, billed=True)
    terminated: list[str] = []
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        args = tuple(argv)
        if "pods" in args and "list" in args:
            rows = [{"id": "unrelated", "name": "someone-else"}]
            raw = canonical_json({"pods": rows, "total_count": 1, "offset": 0, "limit": 100})
        elif "terminate" in args:
            terminated.append(args[args.index("terminate") + 1])
            raw = b""
        elif "disks" in args:
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        elif "wallet" in args:
            raw = _wallet(29.5, billed=True)
        else:
            raw = b""
        return lifecycle.CommandResult(args, 0, raw, b"")
    object.__setattr__(context, "run", run)
    ok, evidence, errors = prime_owner.cleanup(owner, _wallet_snapshot(_wallet(30.0, billed=False)))
    assert not ok and not terminated
    assert "pods:global_inventory_not_empty" in errors
    assert evidence["disks_after_count"] == 0 and evidence["wallet_after"] is None
    assert any(item.startswith("wallet:") for item in errors)
def test_ambiguous_termination_is_dispatched_at_most_once_and_remains_terminal(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _Transport([]), _Commands(tmp_path))
    root = exclusive_runtime_root(tmp_path)
    owner = prime_owner._Lifecycle(context, root, 1.0)
    owner.create_dispatched = True
    owner.known_pod_ids.add("ambiguous-pod")
    termination_calls = 0
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        nonlocal termination_calls
        args = tuple(argv)
        if "pods" in args and "list" in args:
            rows = [{"id": "ambiguous-pod", "name": owner.pod_name}]
            raw = canonical_json({"pods": rows, "total_count": 1, "offset": 0, "limit": 100})
            code = 0
        elif "terminate" in args:
            termination_calls += 1
            raw, code = b"", 1
        elif "disks" in args:
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
            code = 0
        elif "wallet" in args:
            raw, code = _wallet(29.5, billed=True), 0
        else:
            raw, code = b"", 0
        return lifecycle.CommandResult(args, code, raw, b"")
    object.__setattr__(context, "run", run)
    ok, _evidence, errors = prime_owner.cleanup(
        owner, _wallet_snapshot(_wallet(30.0, billed=False))
    )
    assert not ok and termination_calls == 1
    assert any(item.startswith("terminate:") for item in errors)
def test_cleanup_waits_for_nonzero_settled_billing(tmp_path: Path) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path, billing_delay=3)
    result = _run_fixture(_context(tmp_path, transport, commands))
    assert result.state == "completed"
    assert commands.wallet_after_polls == 4
def test_cleanup_exact_maximum_terminates_trusted_and_eight_late_ids(
    tmp_path: Path,
) -> None:
    transport = _Transport([])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    owner = prime_owner._Lifecycle(context, exclusive_runtime_root(tmp_path), 1.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000
    wallet_requests = 0
    def delayed_wallet(_offset: int, _ordinal: int) -> bytes:
        nonlocal wallet_requests
        wallet_requests += 1
        poll = (wallet_requests + 1) // 2
        return (
            _wallet(29.5, billed=True, resource_id="trusted")
            if poll == 12
            else _wallet(30.0, billed=False)
        )

    transport.wallet_handler = delayed_wallet
    owner.trusted_pod_id = "trusted"
    owner.known_pod_ids.add("trusted")
    terminated: list[str] = []
    pod_polls = 0
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        nonlocal pod_polls
        args = tuple(argv)
        if "pods" in args and "list" in args:
            pod_polls += 1
            rows = (
                [{"id": f"late-{index}", "name": owner.pod_name} for index in range(8)]
                if pod_polls == 1
                else []
            )
            raw = canonical_json(
                {"pods": rows, "total_count": len(rows), "offset": 0, "limit": 100}
            )
        elif "terminate" in args:
            terminated.append(args[args.index("terminate") + 1])
            raw = b""
        elif "disks" in args:
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        else:
            raw = b""
        return lifecycle.CommandResult(args, 0, raw, b"")
    object.__setattr__(context, "run", run)
    ok, _evidence, errors = prime_owner.cleanup(
        owner, _wallet_snapshot(_wallet(30.0, billed=False))
    )
    assert ok and not errors
    assert sorted(terminated) == [*[f"late-{index}" for index in range(8)], "trusted"]
    assert owner.cleanup_cli_calls == MAX_CLEANUP_PRIME_CLI_CALLS
    with pytest.raises(RuntimeError, match="cleanup CLI call budget"):
        owner.list_disks(cleanup=True)
def test_cleanup_has_separate_deadline_after_operational_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prime_owner, "MAX_TERMINATION_POLLS", 1)
    clock = [21_601.0]
    context = _context(tmp_path, _Transport([]), _Commands(tmp_path))
    object.__setattr__(context, "monotonic", lambda: clock[0])
    owner = prime_owner._Lifecycle(context, exclusive_runtime_root(tmp_path), 0.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000
    owner.known_pod_ids.add("owned")
    transport = cast(_Transport, context.client.client)
    transport.wallet_handler = lambda _offset, _ordinal: _wallet(
        29.5, billed=True, resource_id="owned"
    )
    terminated: list[str] = []
    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        args = tuple(argv)
        if "pods" in args and "list" in args:
            raw = canonical_json({"pods": [], "total_count": 0, "offset": 0, "limit": 100})
        elif "terminate" in args:
            terminated.append(args[args.index("terminate") + 1])
            raw = b""
        elif "disks" in args:
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        elif "wallet" in args:
            raw = _wallet(29.5, billed=True)
        else:
            raise AssertionError("non-cleanup command crossed the operational deadline")
        return lifecycle.CommandResult(args, 0, raw, b"")
    object.__setattr__(context, "run", run)
    ok, _evidence, errors = prime_owner.cleanup(
        owner, _wallet_snapshot(_wallet(30.0, billed=False))
    )
    assert ok and not errors and terminated == ["owned"]
    assert owner.cleanup_deadline == 21_601.0 + CLEANUP_TIMEOUT_SECONDS
def test_cleanup_deadline_bounds_slow_commands_and_sleeps(tmp_path: Path) -> None:
    clock = [21_601.0]
    timeouts: list[float] = []
    phases: list[str] = []
    transport = _Transport([])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    object.__setattr__(context, "monotonic", lambda: clock[0])
    object.__setattr__(
        context, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    owner = prime_owner._Lifecycle(context, exclusive_runtime_root(tmp_path), 0.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000

    def wallet_phase(_offset: int, _ordinal: int) -> bytes:
        phases.append("wallet")
        return _wallet(30.0, billed=False)

    transport.wallet_handler = wallet_phase

    def run(argv: Sequence[str], _input: bytes | None, timeout: float) -> lifecycle.CommandResult:
        args = tuple(argv)
        timeouts.append(timeout)
        phases.append("disks" if "disks" in args else "wallet" if "wallet" in args else "pods")
        clock[0] += timeout
        if "disks" in args:
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        elif "wallet" in args:
            raw = _wallet(30.0, billed=False)
        else:
            raw = canonical_json({"pods": [], "total_count": 0, "offset": 0, "limit": 100})
        return lifecycle.CommandResult(args, 0, raw, b"")

    object.__setattr__(context, "run", run)
    ok, evidence, errors = prime_owner.cleanup(owner, _wallet_snapshot(_wallet(30.0, billed=False)))
    assert not ok and errors
    assert clock[0] <= 21_601.0 + CLEANUP_TIMEOUT_SECONDS
    assert all(0 < timeout <= COMMAND_TIMEOUT_SECONDS for timeout in timeouts)
    assert evidence["pods_after_count"] == 0
    assert "disks" in phases and "wallet" in phases


def test_cleanup_terminates_known_id_before_failing_inventory_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prime_owner, "MAX_TERMINATION_POLLS", 2)
    transport = _Transport([])
    context = _context(tmp_path, transport, _Commands(tmp_path))
    owner = prime_owner._Lifecycle(context, exclusive_runtime_root(tmp_path), 1.0)
    owner.create_dispatched = True
    owner.create_dispatch_epoch = 1_000
    owner.known_pod_ids.add("trusted")
    order: list[str] = []
    def wallet_after(_offset: int, _ordinal: int) -> bytes:
        order.append("wallet")
        return _wallet(29.5, billed=True, resource_id="trusted")
    transport.wallet_handler = wallet_after

    def run(argv: Sequence[str], _input: bytes | None, _timeout: float) -> lifecycle.CommandResult:
        args = tuple(argv)
        if "terminate" in args:
            order.append(f"terminate:{args[args.index('terminate') + 1]}")
            raw = b""
        elif "pods" in args and "list" in args:
            order.append("pods-list")
            raise subprocess.TimeoutExpired(args, _timeout)
        elif "disks" in args:
            order.append("disks")
            raw = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
        else:
            raise AssertionError("unexpected cleanup command")
        return lifecycle.CommandResult(args, 0, raw, b"")
    object.__setattr__(context, "run", run)
    ok, evidence, errors = prime_owner.cleanup(owner, _wallet_snapshot(_wallet(30.0, billed=False)))
    assert not ok and order[0] == "terminate:trusted"
    assert order.count("terminate:trusted") == 1 and "pods-list" in order
    assert "disks" in order and "wallet" in order
    assert evidence["pods_after_count"] is None and evidence["disks_after_count"] == 0
    assert evidence["wallet_after"] is not None
    assert any(item == "pods:TimeoutExpired" for item in errors)


def test_full_owner_terminal_follows_preinventory_terminate_when_cleanup_lists_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prime_owner, "MAX_TERMINATION_POLLS", 2)
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    context = _context(tmp_path, transport, commands)
    real_run = context.run
    cleanup_order: list[str] = []
    def run(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        args = tuple(argv)
        if commands.terminated and "pods" in args and "list" in args:
            cleanup_order.append("pods-list")
            raise subprocess.TimeoutExpired(args, timeout)
        result = real_run(argv, input_bytes, timeout)
        if "terminate" in args:
            cleanup_order.append(f"terminate:{args[args.index('terminate') + 1]}")
        elif "disks" in args and commands.terminated:
            cleanup_order.append("disks")
        return result
    object.__setattr__(context, "run", run)
    result = _run_fixture(context)
    assert result.state == "failed_terminal" and cleanup_order[0] == "terminate:pod-fixture"
    assert cleanup_order.count("terminate:pod-fixture") == 1
    assert "pods-list" in cleanup_order and "disks" in cleanup_order
    assert transport.wallet_post_zero_requests > 0
    terminal = json.loads((tmp_path / EVIDENCE_ROOT / "terminal.json").read_bytes())
    assert terminal["state"] == "failed_terminal"
    assert terminal["cleanup_failures"] and terminal["wallet_api_call_count"] >= 4


def test_permanent_zero_billing_is_terminal(tmp_path: Path) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path, billing_delay=10_000)
    result = _run_fixture(_context(tmp_path, transport, commands))
    assert result.state == "failed_terminal"
    terminal = json.loads((tmp_path / EVIDENCE_ROOT / "terminal.json").read_bytes())
    assert terminal["cleanup_proven"] is False
    assert terminal["cleanup_failures"]


def test_remote_failure_recovers_junit_and_still_tears_down(tmp_path: Path) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path, remote_returncode=7)
    result = _run_fixture(_context(tmp_path, transport, commands))
    assert result.state == "failed_terminal" and not result.tests_passed
    root = tmp_path / EVIDENCE_ROOT
    assert (root / "pytest.xml").is_file() and (root / "remote-status.json").is_file()
    assert result.cleanup_proven and commands.terminated
    assert sum(call[0].lower().endswith("ssh.exe") for call in commands.invocations) == 1


@_parametrize("failed_name", ["cleanup.json", "command-records.json"])
def test_publication_failures_do_not_bypass_teardown_or_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    real_publish = publish_once
    def fail_selected(path: Path, raw: bytes) -> None:
        if path.name == failed_name:
            raise OSError("fixture publication failure")
        real_publish(path, raw)

    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_test_one_shot_lifecycle_v2.publish_once",
        fail_selected,
    )
    commands = _Commands(tmp_path)
    result = _run_fixture(_context(tmp_path, _Transport([_item(1)]), commands))
    assert result.state == "failed_terminal" and commands.terminated
    terminal = json.loads((tmp_path / EVIDENCE_ROOT / "terminal.json").read_bytes())
    assert terminal["publication_failures"] == ["OSError"]


def test_terminal_signing_or_envelope_publication_failure_gets_fixed_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_sign = lifecycle._sign
    def fail_terminal_sign(
        context: _RuntimeContext,
        payload: bytes,
        namespace: str,
        *,
        timeout: int,
    ) -> bytes:
        if namespace == TERMINAL_NAMESPACE:
            raise OSError("fixture signing failure")
        return real_sign(context, payload, namespace, timeout=timeout)

    monkeypatch.setattr(lifecycle, "_sign", fail_terminal_sign)
    result = _run_fixture(
        _context(tmp_path, _Transport([_item(1)]), _Commands(tmp_path))
    )
    root = tmp_path / EVIDENCE_ROOT
    assert result.state == "failed_terminal" and result.terminal_sha256 is None
    assert (root / "terminal-publication-failure.json").is_file()
    assert not (root / "terminal.json").exists()


def test_terminal_envelope_publication_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = publish_once

    def fail_envelope(path: Path, raw: bytes) -> None:
        if path.name == "terminal-envelope.json":
            raise OSError("fixture terminal envelope failure")
        real_publish(path, raw)

    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_test_one_shot_lifecycle_v2.publish_once",
        fail_envelope,
    )
    result = _run_fixture(
        _context(tmp_path, _Transport([_item(1)]), _Commands(tmp_path))
    )
    root = tmp_path / EVIDENCE_ROOT
    assert result.state == "failed_terminal" and result.terminal_sha256 is None
    assert (root / "terminal.json").is_file()
    assert (root / "terminal-publication-failure.json").is_file()


def test_handoff_sign_timeout_enters_cleanup_and_terminal_without_retry(
    tmp_path: Path,
) -> None:
    commands = _Commands(tmp_path)
    context = _context(tmp_path, _Transport([_item(1)]), commands)
    real_run = context.run
    signing_calls = 0
    def timeout_handoff(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        nonlocal signing_calls
        if Path(argv[0]).name.lower() == "ssh-keygen.exe":
            signing_calls += 1
            if signing_calls == 2:
                raise subprocess.TimeoutExpired(argv, timeout)
        return real_run(argv, input_bytes, timeout)
    object.__setattr__(context, "run", timeout_handoff)
    result = _run_fixture(context)
    root = tmp_path / EVIDENCE_ROOT
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert result.state == "failed_terminal" and result.create_dispatched
    assert result.cleanup_proven and commands.terminated and signing_calls == 3
    assert terminal["primary_failure"] == "TimeoutExpired"
    assert [
        method for method, _url, _payload in cast(_Transport, context.client.client).calls
    ].count("POST") == 1


def test_terminal_sign_timeout_is_bounded_and_publishes_fixed_fallback(
    tmp_path: Path,
) -> None:
    commands = _Commands(tmp_path)
    context = _context(tmp_path, _Transport([_item(1)]), commands)
    real_run = context.run
    signing_calls = 0
    def timeout_terminal(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        nonlocal signing_calls
        if Path(argv[0]).name.lower() == "ssh-keygen.exe":
            signing_calls += 1
            if signing_calls == 3:
                raise subprocess.TimeoutExpired(argv, timeout)
        return real_run(argv, input_bytes, timeout)
    object.__setattr__(context, "run", timeout_terminal)
    result = _run_fixture(context)
    root = tmp_path / EVIDENCE_ROOT
    assert result.state == "failed_terminal" and result.cleanup_proven
    assert commands.terminated and signing_calls == 3
    assert (root / "terminal-publication-failure.json").is_file()
    assert not (root / "terminal-envelope.json").exists()


def test_subprocess_timeout_preserves_failure_and_runs_cleanup(tmp_path: Path) -> None:
    commands = _Commands(tmp_path)
    original = commands.__call__
    def timeout_remote(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        if Path(argv[0]).name.lower() == "ssh.exe":
            raise subprocess.TimeoutExpired(argv, timeout)
        return original(argv, input_bytes, timeout)
    context = _context(tmp_path, _Transport([_item(1)]), commands)
    object.__setattr__(context, "run", timeout_remote)
    result = _run_fixture(context)
    assert result.state == "failed_terminal" and result.cleanup_proven
    root = tmp_path / EVIDENCE_ROOT
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert terminal["primary_failure"] == "TimeoutExpired"
    assert terminal["recovery_failures"] == []
    assert (root / "gpu-facts.json").is_file()
    assert (root / "pytest.xml").is_file()
    assert (root / "remote-status.json").is_file()


def test_generic_remote_runner_exception_recovers_once_then_cleans_up(
    tmp_path: Path,
) -> None:
    commands = _Commands(tmp_path)
    original = commands.__call__
    def fail_remote(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        if Path(argv[0]).name.lower() == "ssh.exe" and "bash" in argv:
            raise OSError("fixture remote runner failure")
        return original(argv, input_bytes, timeout)
    context = _context(tmp_path, _Transport([_item(1)]), commands)
    object.__setattr__(context, "run", fail_remote)
    result = _run_fixture(context)
    root = tmp_path / EVIDENCE_ROOT
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert result.state == "failed_terminal" and result.cleanup_proven
    assert terminal["primary_failure"] == "OSError"
    assert terminal["recovery_failures"] == []
    assert (
        sum(
            Path(call[0]).name.lower() == "scp.exe"
            and any("gpu-facts.json" in item for item in call)
            for call in commands.invocations
        )
        == 1
    )


def test_expired_operational_deadline_still_cleans_and_publishes_terminal(
    tmp_path: Path,
) -> None:
    commands = _Commands(tmp_path)
    original = commands.__call__
    clock = [1.0]

    def expire_after_remote(
        argv: Sequence[str], input_bytes: bytes | None, timeout: float
    ) -> lifecycle.CommandResult:
        result = original(argv, input_bytes, timeout)
        if Path(argv[0]).name.lower() == "ssh.exe" and "bash" in argv:
            clock[0] = 21_602.0
        return result

    context = _context(tmp_path, _Transport([_item(1)]), commands)
    object.__setattr__(context, "monotonic", lambda: clock[0])
    object.__setattr__(context, "run", expire_after_remote)
    result = _run_fixture(context)
    terminal = json.loads((tmp_path / EVIDENCE_ROOT / "terminal.json").read_bytes())
    assert result.state == "failed_terminal" and result.cleanup_proven
    assert terminal["primary_failure"] == "TimeoutError"
    assert commands.terminated


def test_full_source_free_lifecycle_recovers_junit_then_cleans_up(
    tmp_path: Path,
) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    context = _context(tmp_path, transport, commands)
    result = _run_fixture(context)
    assert result.state == "completed"
    assert result.tests_passed and result.cleanup_proven and result.create_dispatched
    assert [method for method, _url, _payload in transport.calls].count("POST") == 1
    availability = [
        url
        for method, url, _payload in transport.calls
        if method == "GET" and url != prime_owner.WALLET_API_ENDPOINT
    ]
    wallet_calls = [
        url
        for method, url, _payload in transport.calls
        if method == "GET" and url == prime_owner.WALLET_API_ENDPOINT
    ]
    assert len(availability) == 2 and len(wallet_calls) == 4
    root = tmp_path / EVIDENCE_ROOT
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert terminal["tests_passed"] is True
    assert terminal["cleanup_proven"] is True
    assert "junit" in terminal["evidence_dag"]
    assert "cleanup" in terminal["evidence_dag"]
    assert commands.terminated
    assert not any("pods create" in " ".join(call) for call in commands.invocations)
    assert evidence.verify_terminal_evidence(root, context.identity)["state"] == "completed"
    sanitized = b"".join(
        (root / name).read_bytes()
        for name in ("command-journal.jsonl", "wallet-before.json", "cleanup.json", "terminal.json")
    )
    for secret in (
        b"team-fixture",
        b"wallet-fixture",
        b"billing-fixture",
        b"pod-fixture",
        b"fixture-provider",
    ):
        assert secret not in sanitized
    assert sha256_bytes(b"team-fixture").encode() in sanitized
    (root / "pytest.xml").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        evidence.verify_terminal_evidence(root, context.identity)


def test_cli_source_proves_zero_disk_cli_path_is_forbidden() -> None:
    source = Path(v5._site_packages()) / prime_owner.PODS_COMMAND_OWNER
    text = source.read_text(encoding="utf-8")
    assert "if not disk_size:" in text
    assert "availability_client.get()" in text


def test_assessment_ttl_is_enforced_before_create(tmp_path: Path) -> None:
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    context = _context(tmp_path, transport, commands)
    epochs = iter((1_000, 1_000, 1_000 + ASSESSMENT_TTL_SECONDS + 1))
    object.__setattr__(context, "now", lambda: next(epochs))
    result = _run_fixture(context)
    assert result.state == "failed_terminal"
    assert not result.create_dispatched
    assert not any(call[0] == "POST" for call in transport.calls)


def test_completed_topology_binds_two_identical_keyscan_outcomes(tmp_path: Path) -> None:
    _context_value, root = _completed_with_history(tmp_path / "completed")
    summary = prime_owner.replay_command_journal(
        (root / "command-records.json").read_bytes(),
        (root / "command-journal.jsonl").read_bytes(),
    )
    digest = sha256_bytes((root / "known-hosts.txt").read_bytes())
    assert summary.ssh_keyscan_stdout_sha256s == (digest, digest)


def test_runtime_authority_excludes_model_science_source_and_training() -> None:
    for key in (
        "model_calls_authorized",
        "provider_calls_authorized",
        "science_authorized",
        "source_access_authorized",
        "training_campaign_authorized",
    ):
        assert RUNTIME_AUTHORITY[key] is False


def _rewrite_coherent_wire_chain(
    context: _RuntimeContext,
    root: Path,
    terminal: dict[str, Any],
    blob: str,
    algorithm: str = "ssh-rsa",
) -> None:
    known = f"[8.8.8.8]:2222 {algorithm} {blob}\n".encode()
    digest = sha256_bytes(known)
    handoff = cast(dict[str, Any], json.loads((root / "handoff.json").read_bytes()))
    handoff["ssh"]["known_hosts_sha256"] = digest
    _replace_bound_bytes(root, terminal, "known-hosts", known)
    records = _journal_records(root)
    for record in records:
        if record["phase"] == "outcome" and cast(str, record["operation"]).startswith(
            "ssh-keyscan.exe "
        ):
            details = cast(dict[str, object], record["details"])
            details["stdout_sha256"], details["stdout_bytes"] = digest, len(known)
    _bind_journal(root, terminal, records)
    raw = canonical_json(handoff)
    signature = lifecycle._sign(context, raw, HANDOFF_NAMESPACE, timeout=30)
    envelope = evidence.signed_envelope(
        raw, signature, HANDOFF_NAMESPACE, context.identity, authority=READINESS_AUTHORITY
    )
    for name, value in (
        ("handoff", raw),
        ("handoff-signature", signature),
        ("handoff-envelope", envelope),
    ):
        _replace_bound_bytes(root, terminal, name, value)
    _resign_terminal(context, root, terminal)


@pytest.mark.parametrize("algorithm,blob", [
    ("ssh-rsa", "Z2FyYmFnZQ=="),
    ("ssh-ed25519", "AAAAC3NzaC1lZDI1NTE5AAAAHwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM="),
    ("ssh-rsa", "AAAAB3NzaC1yc2EAAAADAQABAAAAIAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"),
    ("ssh-rsa", "/////w=="),
    ("ssh-rsa", "AAAAB3NzaC1yc2EAAACAAAEBAAAAIAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"),
    ("ssh-rsa", "AAAAB3NzaC1yc2EAAAADAQABAAAAIAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAA=="),
    (
        "ecdsa-sha2-nistp256",
        "AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGsX0fLhLEJH+Lzm5WOkQPJ3A32B"
        "LeszoPShOUXYmMKWT+NC4v4af5uO5+tKfA+eFivOM1drMV7Oy7ZAaDe/UfQ=",
    ),
    (
        "ssh-rsa",
        "AAAAB3NzaC1yc2EAAAADAQABAAAAgAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB",
    ),
])
def test_resigned_terminal_rejects_ssh_wire_key_mutations(
    tmp_path: Path, algorithm: str, blob: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, root = _completed_with_history(tmp_path / "wire")
    terminal = json.loads((root / "terminal.json").read_bytes())
    _rewrite_coherent_wire_chain(context, root, terminal, blob, algorithm)
    calls = 0
    original = handoff_owner._validate_ssh_key_blob

    def spy(blob_bytes: bytes, algorithm: str) -> None:
        nonlocal calls
        calls += 1
        original(blob_bytes, algorithm)

    monkeypatch.setattr(handoff_owner, "_validate_ssh_key_blob", spy)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
    assert calls == 1
