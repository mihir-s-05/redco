"""Source-free tests for the signed Stage D Prime capacity monitor."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar, cast

import pytest

from redco.analysis import stage_d_v13_prime_capacity_monitor_v1 as monitor
from redco.analysis import stage_d_v13_prime_inventory_v5 as v5

ROOT = Path(__file__).parents[1].resolve()
_FixtureFunction = TypeVar("_FixtureFunction", bound=Callable[..., object])
_fixture = cast(Callable[..., Callable[[_FixtureFunction], _FixtureFunction]], pytest.fixture)
_parametrize = cast(
    Callable[..., Callable[[_FixtureFunction], _FixtureFunction]], pytest.mark.parametrize
)


def _item(index: int, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "cloudId": f"fixture-{index}",
        "gpuType": "L40S 48GB",
        "socket": "PCIe",
        "provider": "fixture-provider",
        "dataCenter": "fixture-dc",
        "country": "US",
        "gpuCount": 2,
        "gpuMemory": 96,
        "disk": {
            "minCount": 0,
            "defaultCount": 1250,
            "maxCount": 2000,
            "pricePerUnit": 0.01,
            "step": 1,
            "defaultIncludedInPrice": True,
            "additionalInfo": None,
        },
        "vcpu": None,
        "memory": None,
        "internetSpeed": None,
        "interconnect": None,
        "interconnectType": None,
        "provisioningTime": None,
        "stockStatus": "Available",
        "security": "datacenter",
        "prices": {
            "onDemand": 1.8,
            "communityPrice": 1.64,
            "isVariable": False,
            "currency": "USD",
        },
        "images": [],
        "isSpot": False,
        "prepaidTime": None,
    }
    value.update(changes)
    return value


@dataclass(frozen=True)
class _Response:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _Transport:
    def __init__(
        self,
        endpoint_items: Mapping[str, list[dict[str, object]]],
        *,
        fail_call: int | None = None,
    ) -> None:
        self.endpoint_items = endpoint_items
        self.fail_call = fail_call
        self.calls: list[tuple[str, str, dict[str, object], bool]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object],
        follow_redirects: bool,
    ) -> _Response:
        copied = dict(params)
        self.calls.append((method, url, copied, follow_redirects))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("fixture transport failure")
        endpoint = url.removeprefix(v5.BASE_URL)
        items = self.endpoint_items[endpoint]
        page_value = copied["page"]
        if type(page_value) is not int:
            raise TypeError("fixture page is not an integer")
        page = page_value
        start = (page - 1) * v5.PAGE_SIZE
        body = monitor._canonical(
            {"items": items[start : start + v5.PAGE_SIZE], "totalCount": len(items)}
        )
        return _Response(200, body, {"content-type": "application/json"})


class _Client:
    base_url = v5.BASE_URL
    api_key = "fixture-present-not-persisted"

    def __init__(self, transport: _Transport) -> None:
        self.client = transport


@dataclass
class _Clock:
    wall_value: int = 1_000
    monotonic_value: float = 10.0

    def wall(self) -> int:
        return self.wall_value

    def monotonic(self) -> float:
        value = self.monotonic_value
        self.monotonic_value += 1.0
        return value


@dataclass
class _SequenceClock:
    walls: list[int]
    monotonic_value: float = 10.0

    def wall(self) -> int:
        return self.walls.pop(0)

    def monotonic(self) -> float:
        value = self.monotonic_value
        self.monotonic_value += 1.0
        return value


@_fixture(scope="module")
def signer(tmp_path_factory: pytest.TempPathFactory) -> monitor._Signer:
    root = tmp_path_factory.mktemp("monitor-signing")
    key = root / "id_rsa"
    v5._ssh_keygen(["-q", "-t", "rsa", "-b", "2048", "-N", "", "-f", str(key)])
    public = key.with_name("id_rsa.pub").read_text(encoding="ascii").split()
    key_type, key_base64 = public[:2]
    fingerprint = (
        v5._ssh_keygen(
            ["-lf", "-", "-E", "sha256"],
            input_bytes=f"{key_type} {key_base64}\n".encode("ascii"),
        )
        .decode("ascii")
        .split()[1]
    )
    allowed = f"fixture {key_type} {key_base64}\n".encode("ascii")
    identity = monitor._SigningIdentity(
        "fixture", key_type, key_base64, fingerprint, monitor._sha256(allowed)
    )
    return monitor._Signer(identity, key)


def _context(
    tmp_path: Path,
    signer: monitor._Signer,
    transport: _Transport,
    *,
    clock: monitor._Clock | None = None,
    errors: tuple[type[BaseException], ...] = (RuntimeError,),
    free_bytes: int = 1 << 40,
) -> monitor._HeartbeatContext:
    return monitor._HeartbeatContext(
        repository=ROOT,
        layout=monitor.ObservationLayout(tmp_path / "monitor"),
        signer=signer,
        checkpoint={"commit": "c" * 40, "tree": "d" * 40, "state": "fixture"},
        capture_owners={"fixture": "authenticated"},
        openssh_executable={"fixture": "authenticated"},
        client_factory=lambda: _Client(transport),
        transport_errors=errors,
        clock=clock or _Clock(),
        free_bytes=lambda _path: free_bytes,
    )


def _transport(
    first: list[dict[str, object]],
    second: list[dict[str, object]] | None = None,
    *,
    fail_call: int | None = None,
) -> _Transport:
    return _Transport(
        {v5.ENDPOINTS[0]: first, v5.ENDPOINTS[1]: second or []},
        fail_call=fail_call,
    )


def test_authorization_and_frozen_contract_values() -> None:
    assert len(monitor.AUTHORIZATION_TEXT.encode()) == 146
    assert monitor._sha256(monitor.AUTHORIZATION_TEXT.encode()) == monitor.AUTHORIZATION_SHA256
    assert monitor.PARENT_COMMIT == "5684bb67babff19ebffe697661fccdb660527ac4"
    assert monitor.PARENT_TREE == "0dec35f9a4aaa542d18f17a8d139f6ac748ed1af"
    assert monitor.MINIMUM_CADENCE_SECONDS == 300
    assert monitor.MAXIMUM_OBSERVATIONS == 288
    assert monitor.MAXIMUM_WINDOW_SECONDS == 86_400
    assert monitor.AUTHORIZATION_FALSE == v5.AUTHORIZATION_FALSE
    assert monitor.MAXIMUM_VALID_OBSERVATION_BYTES <= monitor.PRECLAIM_FREE_SPACE_RESERVE
    monitor._authenticate_storage_law()


@_parametrize(
    ("status", "stage", "worktree"),
    [
        ("M  external/prime-rl\n", None, None),
        (" D external/prime-rl\n", None, None),
        (" M external/prime-rl\n?? external/prime-rl/escape\n", None, None),
        (" M external/prime-rl\n?? unrelated\n", None, None),
        (None, "160000 " + "0" * 40 + " 0\texternal/prime-rl", None),
        (None, None, "0" * 40),
    ],
)
def test_external_gitlink_mutations_fail_before_monitor_activity(
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
    stage: str | None,
    worktree: str | None,
) -> None:
    porcelain = status or monitor.EXTERNAL_GITLINK_STATUS + "\n"
    indexed = stage or (
        f"{monitor.EXTERNAL_GITLINK_MODE} {monitor.EXTERNAL_GITLINK_OBJECT} "
        f"0\t{monitor.EXTERNAL_GITLINK_PATH}"
    )

    def fake_git(root: Path, *arguments: str) -> str:
        if root == ROOT / monitor.EXTERNAL_GITLINK_PATH:
            return worktree or monitor.EXTERNAL_GITLINK_OBJECT
        if arguments == ("rev-parse", "HEAD"):
            return monitor.PARENT_COMMIT
        if arguments == ("rev-parse", f"{monitor.PARENT_COMMIT}^{{tree}}"):
            return monitor.PARENT_TREE
        if arguments == ("ls-files", "--stage", "--", monitor.EXTERNAL_GITLINK_PATH):
            return indexed
        raise AssertionError((root, arguments))

    monkeypatch.setattr(monitor, "_git", fake_git)
    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_capacity_monitor_v1.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=porcelain),
    )
    with pytest.raises(ValueError):
        monitor.authenticate_monitor_checkpoint(ROOT, precommit=True)
    assert not (ROOT / monitor.MONITOR_ROOT_RELATIVE).exists()


def test_deterministic_contract_and_audit() -> None:
    first = monitor.build_checkpoint_artifacts(ROOT)
    second = monitor.build_checkpoint_artifacts(ROOT)
    assert first == second
    contract = json.loads(first[monitor.CONTRACT_RELATIVE])
    audit = json.loads(first[monitor.AUDIT_RELATIVE])
    assert contract["domain"] == monitor.CONTRACT_DOMAIN
    assert contract["monitor_authorization"]["sha256"] == monitor.AUTHORIZATION_SHA256
    assert contract["live_monitoring_authorized_by_checkpoint"] is False
    assert contract["authorization"] == monitor.AUTHORIZATION_FALSE
    assert audit["domain"] == monitor.AUDIT_DOMAIN
    assert audit["authorization"] == monitor.AUTHORIZATION_FALSE


@_parametrize(
    ("items", "state", "disposition"),
    [
        ([_item(1, isSpot=None)], "observed_no_qualifying_resource", "continue_monitoring"),
        ([_item(1)], "observed_non_authorizing_resource", "qualifying_capacity_found_stop"),
        (
            [_item(1, isSpot=None), _item(2, cloudId="fixture-1", provider="other", isSpot=None)],
            "observed_ambiguous_resources",
            "continue_monitoring",
        ),
    ],
)
def test_real_v5_semantics_drive_monitor_disposition(
    tmp_path: Path,
    signer: monitor._Signer,
    items: list[dict[str, object]],
    state: str,
    disposition: str,
) -> None:
    transport = _transport(items)
    context = _context(tmp_path, signer, transport)
    result = monitor._run_heartbeat(context)
    assert result.state == state
    assert result.disposition == disposition
    assert len(transport.calls) == 2
    assert [call[1].removeprefix(v5.BASE_URL) for call in transport.calls] == list(v5.ENDPOINTS)
    assert all(call[2]["gpu_count"] == "2" and call[3] is False for call in transport.calls)
    ledger = context.layout.ledger_record(1)
    assert ledger.is_file()
    payload, _ = monitor._verify_envelope(
        ledger.read_bytes(),
        domain=monitor.LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload_domain=monitor.LEDGER_PAYLOAD_DOMAIN,
        namespace=monitor.LEDGER_NAMESPACE,
        identity=signer.identity,
    )
    assert payload["formal_state"] == state
    assert payload["resource"] is None
    assert payload["authorization"] == monitor.AUTHORIZATION_FALSE
    assessment = context.layout.artifact(1, "assessment.json").read_bytes()
    value = json.loads(assessment)
    assert set(value) == {
        "schema_version",
        "domain",
        "state",
        "reason",
        "resource",
        "semantic_projection_sha256",
        "counts",
        "authorization",
    }
    for forbidden in (b"cloud_id", b"raw_item", b"fixture-provider", b"fixture-1"):
        assert forbidden not in assessment


def test_full_pagination_and_exact_request_count(tmp_path: Path, signer: monitor._Signer) -> None:
    transport = _transport([_item(index, isSpot=None) for index in range(101)])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.continue_monitoring
    assert len(transport.calls) == 3
    assert [call[2]["page"] for call in transport.calls] == [1, 2, 1]


def test_exact_cadence_restart_and_hash_chain(tmp_path: Path, signer: monitor._Signer) -> None:
    clock = _Clock()
    first_transport = _transport([_item(1, isSpot=None)])
    context = _context(tmp_path, signer, first_transport, clock=clock)
    first = monitor._run_heartbeat(context)
    assert first.observation_ordinal == 1
    clock.wall_value = 1_299
    early_transport = _transport([])
    early_context = _context(tmp_path, signer, early_transport, clock=clock)
    early = monitor._run_heartbeat(early_context)
    assert early.disposition == "cadence_not_elapsed_noop"
    assert early_transport.calls == []
    assert not early_context.layout.observation_dir(2).exists()
    clock.wall_value = 1_300
    second_transport = _transport([_item(2, isSpot=None)])
    second = monitor._run_heartbeat(_context(tmp_path, signer, second_transport, clock=clock))
    assert second.observation_ordinal == 2
    assert second.continue_monitoring
    chain = monitor._validate_chain(_context(tmp_path, signer, second_transport, clock=clock))
    assert chain.next_ordinal == 3


def test_claim_epoch_cadence_exact_boundary(tmp_path: Path, signer: monitor._Signer) -> None:
    first = monitor._run_heartbeat(
        _context(
            tmp_path,
            signer,
            _transport([_item(1, isSpot=None)]),
            clock=_SequenceClock([1_000, 1_010]),
        )
    )
    assert first.next_not_before_epoch == 1_300
    early = monitor._run_heartbeat(
        _context(tmp_path, signer, _transport([]), clock=_SequenceClock([1_299]))
    )
    assert early.disposition == "cadence_not_elapsed_noop"
    assert early.next_not_before_epoch == 1_300
    exact_transport = _transport([_item(2, isSpot=None)])
    exact = monitor._run_heartbeat(
        _context(
            tmp_path,
            signer,
            exact_transport,
            clock=_SequenceClock([1_300, 1_310]),
        )
    )
    assert exact.observation_ordinal == 2
    assert exact_transport.calls


def test_signed_cadence_mutation_fails_before_next_request(
    tmp_path: Path, signer: monitor._Signer
) -> None:
    context = _context(tmp_path, signer, _transport([_item(1, isSpot=None)]))
    monitor._run_heartbeat(context)
    ledger = context.layout.ledger_record(1)
    payload, _ = monitor._verify_envelope(
        ledger.read_bytes(),
        domain=monitor.LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload_domain=monitor.LEDGER_PAYLOAD_DOMAIN,
        namespace=monitor.LEDGER_NAMESPACE,
        identity=signer.identity,
    )
    timing = cast(dict[str, object], payload["timing"])
    timing["next_not_before_epoch"] = 1_301
    payload_raw = monitor._canonical(payload)
    signature = monitor._sign_bytes(signer, payload_raw, monitor.LEDGER_NAMESPACE)
    ledger.write_bytes(
        monitor._envelope(
            domain=monitor.LEDGER_ENVELOPE_DOMAIN,
            state="signed_ledger_record",
            payload=payload_raw,
            signature=signature,
            identity=signer.identity,
            namespace=monitor.LEDGER_NAMESPACE,
        )
    )
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.disposition == "prior_chain_invalid_terminal"
    assert transport.calls == []


def test_clock_rollback_is_terminal_before_claim(tmp_path: Path, signer: monitor._Signer) -> None:
    clock = _Clock()
    context = _context(tmp_path, signer, _transport([_item(1, isSpot=None)]), clock=clock)
    monitor._run_heartbeat(context)
    clock.wall_value = 999
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport, clock=clock))
    assert result.disposition == "clock_rollback_terminal"
    assert transport.calls == []
    assert not context.layout.observation_dir(2).exists()


def test_transport_failure_consumes_one_attempt_and_stops(
    tmp_path: Path, signer: monitor._Signer
) -> None:
    transport = _transport([], fail_call=1)
    context = _context(tmp_path, signer, transport)
    result = monitor._run_heartbeat(context)
    assert not result.continue_monitoring
    assert result.state == "capture_failed_terminal"
    assert result.exit_code == 20
    assert len(transport.calls) == 1
    again_transport = _transport([])
    again = monitor._run_heartbeat(_context(tmp_path, signer, again_transport))
    assert not again.continue_monitoring
    assert again_transport.calls == []


def test_incomplete_claim_after_crash_is_terminal(tmp_path: Path, signer: monitor._Signer) -> None:
    context = _context(tmp_path, signer, _transport([]))
    context.layout.observations.mkdir(parents=True)
    context.layout.ledger.mkdir()
    directory = context.layout.observation_dir(1)
    directory.mkdir()
    (directory / "claim.json").write_bytes(b"{}")
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.disposition == "prior_chain_invalid_terminal"
    assert transport.calls == []
    assert not context.layout.observation_dir(2).exists()


def test_mutated_assessment_or_ledger_signature_stops_chain(
    tmp_path: Path, signer: monitor._Signer
) -> None:
    context = _context(tmp_path, signer, _transport([_item(1, isSpot=None)]))
    monitor._run_heartbeat(context)
    assessment = context.layout.artifact(1, "assessment.json")
    assessment.write_bytes(assessment.read_bytes() + b" ")
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.disposition == "prior_chain_invalid_terminal"
    assert transport.calls == []


def test_signature_domain_and_payload_mutations_reject(
    signer: monitor._Signer,
) -> None:
    payload = monitor._canonical(
        {"schema_version": 1, "domain": monitor.LEDGER_PAYLOAD_DOMAIN, "value": 1}
    )
    signature = monitor._sign_bytes(signer, payload, monitor.LEDGER_NAMESPACE)
    envelope = monitor._envelope(
        domain=monitor.LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload=payload,
        signature=signature,
        identity=signer.identity,
        namespace=monitor.LEDGER_NAMESPACE,
    )
    monitor._verify_envelope(
        envelope,
        domain=monitor.LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload_domain=monitor.LEDGER_PAYLOAD_DOMAIN,
        namespace=monitor.LEDGER_NAMESPACE,
        identity=signer.identity,
    )
    value = json.loads(envelope)
    value["public_identity"]["namespace"] = monitor.OBSERVATION_NAMESPACE
    with pytest.raises(ValueError, match="projection"):
        monitor._verify_envelope(
            monitor._canonical(value),
            domain=monitor.LEDGER_ENVELOPE_DOMAIN,
            state="signed_ledger_record",
            payload_domain=monitor.LEDGER_PAYLOAD_DOMAIN,
            namespace=monitor.LEDGER_NAMESPACE,
            identity=signer.identity,
        )


def test_storage_limit_fails_before_claim(tmp_path: Path, signer: monitor._Signer) -> None:
    transport = _transport([])
    context = _context(tmp_path, signer, transport, free_bytes=0)
    result = monitor._run_heartbeat(context)
    assert result.disposition == "storage_limit_terminal"
    assert transport.calls == []
    assert not context.layout.observation_dir(1).exists()


def test_first_observation_limit_stops_at_frozen_bound(
    tmp_path: Path, signer: monitor._Signer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor, "MAXIMUM_OBSERVATIONS", 1)
    result = monitor._run_heartbeat(_context(tmp_path, signer, _transport([_item(1, isSpot=None)])))
    assert result.disposition == "monitor_window_exhausted_no_capacity"
    assert not result.continue_monitoring


def test_lock_hardlink_rejected_without_touching_victim(tmp_path: Path) -> None:
    layout = monitor.ObservationLayout(tmp_path / "monitor")
    layout.root.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim-bytes")
    os.link(victim, layout.lock)
    before = (victim.read_bytes(), victim.stat().st_nlink, victim.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="lock is linked"), monitor._monitor_lock(layout):
        raise AssertionError("hardlinked lock was acquired")
    assert (victim.read_bytes(), victim.stat().st_nlink, victim.stat().st_mtime_ns) == before


def test_lock_reparse_and_parent_alias_reject_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = monitor.ObservationLayout(tmp_path / "monitor")
    original = monitor._is_link_or_reparse
    monkeypatch.setattr(
        monitor,
        "_is_link_or_reparse",
        lambda path: path == layout.root.parent or original(path),
    )
    with pytest.raises(ValueError, match="ancestor"), monitor._monitor_lock(layout):
        raise AssertionError("aliased parent lock was acquired")
    assert not layout.root.exists()


@_parametrize("artifact_name", ["terminal-auth.json", "ledger"])
def test_signed_evidence_hardlinks_fail_before_next_request(
    tmp_path: Path, signer: monitor._Signer, artifact_name: str
) -> None:
    context = _context(tmp_path, signer, _transport([_item(1, isSpot=None)]))
    monitor._run_heartbeat(context)
    target = (
        context.layout.ledger_record(1)
        if artifact_name == "ledger"
        else context.layout.artifact(1, artifact_name)
    )
    victim = tmp_path / f"{artifact_name}.victim"
    victim.write_bytes(target.read_bytes())
    target.unlink()
    os.link(victim, target)
    before = (victim.read_bytes(), victim.stat().st_nlink, victim.stat().st_mtime_ns)
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.disposition == "prior_chain_invalid_terminal"
    assert transport.calls == []
    assert (victim.read_bytes(), victim.stat().st_nlink, victim.stat().st_mtime_ns) == before


@_parametrize("artifact_name", ["terminal-auth.json", "ledger"])
def test_signed_evidence_reparse_paths_fail_before_next_request(
    tmp_path: Path,
    signer: monitor._Signer,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    context = _context(tmp_path, signer, _transport([_item(1, isSpot=None)]))
    monitor._run_heartbeat(context)
    target = (
        context.layout.ledger_record(1)
        if artifact_name == "ledger"
        else context.layout.artifact(1, artifact_name)
    )
    original = monitor._is_link_or_reparse
    monkeypatch.setattr(
        monitor,
        "_is_link_or_reparse",
        lambda path: path == target or original(path),
    )
    transport = _transport([])
    result = monitor._run_heartbeat(_context(tmp_path, signer, transport))
    assert result.disposition == "prior_chain_invalid_terminal"
    assert transport.calls == []


def test_v5_declared_call_allowlist_matches_static_calls() -> None:
    tree = ast.parse((ROOT / monitor.OWNER_RELATIVE).read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "v5"
    }
    assert calls == set(monitor.V5_PRIMITIVE_ALLOWLIST)


def test_fixed_runner_has_no_scientific_or_path_arguments() -> None:
    text = (ROOT / monitor.RUNNER_RELATIVE).read_text(encoding="utf-8")
    assert "sys.argv[1:]" in text
    assert "argparse" not in text
    assert "wallet" not in text.casefold()
    assert "provision" not in text.casefold()


def test_public_runner_precommit_failure_is_sanitized_exit_20() -> None:
    tool_python = Path(os.environ["APPDATA"]) / "uv" / "tools" / "prime" / "Scripts" / "python.exe"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    monitor_root = ROOT / monitor.MONITOR_ROOT_RELATIVE
    before = monitor_root.exists()
    result = subprocess.run(
        [str(tool_python), str(ROOT / monitor.RUNNER_RELATIVE)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 20
    assert result.stderr == ""
    value = json.loads(result.stdout)
    assert value["state"] == "runner_failure_terminal"
    assert value["authorization"] == monitor.AUTHORIZATION_FALSE
    assert monitor._canonical(value).decode() + "\n" == result.stdout
    assert monitor_root.exists() is before


def test_real_prime_tool_preclaim_twice_has_zero_request_and_zero_claim() -> None:
    if sys.platform != "win32":
        raise AssertionError("the reviewed monitor runtime is Windows-only")
    tool_python = Path(os.environ["APPDATA"]) / "uv" / "tools" / "prime" / "Scripts" / "python.exe"
    uv = Path.home() / ".local" / "bin" / "uv.exe"
    code = """
import os,socket
from pathlib import Path
from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
os.environ['PATH']=str(Path.home()/'.local'/'bin')+os.pathsep+os.environ['PATH']
owners=v5.authenticate_installed_capture_owners()
v5._authenticate_config_paths()
identity=v5._load_terminal_signing_identity()
v5.authenticate_approved_openssh_executable()
v5._authenticate_operator_key(Path.home()/'.ssh'/'id_rsa',identity)
httpx=v5._load_httpx_module(); calls=[]
def blocked(*args,**kwargs): calls.append(1); raise AssertionError('request forbidden')
old=httpx.Client.request; httpx.Client.request=blocked
try:
 client=v5._construct_api_client(); assert client.base_url==v5.BASE_URL and client.api_key
 assert v5._httpx_request_error_types()==(httpx.RequestError,); client.client.close()
finally: httpx.Client.request=old
assert calls==[]
print('PRECLAIM_OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    fixed_root = ROOT / monitor.MONITOR_ROOT_RELATIVE
    before = fixed_root.exists()
    for _ in range(2):
        result = subprocess.run(
            [
                str(uv),
                "run",
                "--no-project",
                "--offline",
                "--python",
                str(tool_python),
                "python",
                "-c",
                code,
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == "PRECLAIM_OK"
    assert fixed_root.exists() is before


def test_two_process_lock_overlap_is_nonblocking(tmp_path: Path) -> None:
    layout = monitor.ObservationLayout(tmp_path / "lock-root")
    code = """
import sys,time
from pathlib import Path
from redco.analysis.stage_d_v13_prime_capacity_monitor_v1 import ObservationLayout,_monitor_lock
layout=ObservationLayout(Path(sys.argv[1]))
with _monitor_lock(layout) as acquired:
 print('LOCKED' if acquired else 'OVERLAP',flush=True)
 if acquired: time.sleep(3)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    first = subprocess.Popen(
        [sys.executable, "-c", code, str(layout.root)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert first.stdout is not None
    assert first.stdout.readline().strip() == "LOCKED"
    second = subprocess.run(
        [sys.executable, "-c", code, str(layout.root)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert second.stdout.strip() == "OVERLAP"
    first.wait(timeout=10)
    assert first.returncode == 0
