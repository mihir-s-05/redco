"""Adversarial source-free tests for Prime one-shot signed evidence."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_evidence_v2 as evidence
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis import stage_d_v13_prime_test_one_shot_prime_v2 as prime_owner
from redco.analysis import stage_d_v13_prime_test_one_shot_wallet_v2 as wallet_owner
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ASSESSMENT_NAMESPACE,
    AUTHORIZATION_PATH,
    EVIDENCE_ROOT,
    HANDOFF_NAMESPACE,
    OPENSSH_EXECUTABLES,
    PODS_CREATE_ENDPOINT,
    READINESS_AUTHORITY,
    TERMINAL_NAMESPACE,
    TEST_NODES,
    canonical_json,
    sha256_bytes,
)

P = ParamSpec("P")
R = TypeVar("R")


def _parametrize(
    argnames: str | Sequence[str], argvalues: Sequence[object]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return cast(
        Callable[[Callable[P, R]], Callable[P, R]],
        pytest.mark.parametrize(argnames, argvalues),
    )


def _item(index: int, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "cloudId": f"cloud-{index}",
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
            "onDemand": 1.80,
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
    def __init__(self, items: list[dict[str, object]], *, ambiguous_create: bool = False) -> None:
        self.items = items
        self.ambiguous_create = ambiguous_create
        self.calls: list[tuple[str, str, object]] = []
        self.commands: _Commands | None = None
        self.wallet_handler: Callable[[int, int], bytes] | None = None
        self.wallet_requests = 0
        self.wallet_post_zero_requests = 0
    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append((method, url, kwargs.get("json", kwargs.get("params"))))
        if method == "POST":
            payload = cast(dict[str, Any], kwargs["json"])
            assert url == PODS_CREATE_ENDPOINT
            assert cast(dict[str, Any], payload["pod"])["diskSize"] == 0
            assert payload["disks"] == []
            if self.ambiguous_create:
                raise OSError("fixture ambiguous create")
            return _Response(
                201,
                canonical_json(
                    {
                        "id": "pod-fixture",
                        "name": cast(dict[str, Any], payload["pod"])["name"],
                        "gpuName": "L40S 48GB",
                        "gpuCount": 2,
                        "status": "INSTALLING",
                        "createdAt": "fixture",
                        "providerType": "fixture-provider",
                    }
                ),
                {"content-type": "application/json"},
            )
        if url == prime_owner.WALLET_API_ENDPOINT:
            params = cast(dict[str, object], kwargs["params"])
            assert params["limit"] == 100 and params.get("teamId") == "team-fixture"
            offset = cast(int, params["offset"])
            self.wallet_requests += 1
            if self.wallet_handler is not None:
                raw = self.wallet_handler(offset, self.wallet_requests)
            else:
                commands = self.commands
                assert commands is not None
                if commands.terminated and offset == 0:
                    self.wallet_post_zero_requests += 1
                    commands.wallet_after_polls = (self.wallet_post_zero_requests + 1) // 2
                settled = (
                    commands.terminated
                    and commands.wallet_after_polls > commands.billing_delay
                )
                raw = _wallet(29.5 if settled else 30.0, billed=settled)
            return _Response(200, raw, {"content-type": "application/json"})
        endpoint = url.removeprefix(v5.BASE_URL)
        items = self.items if endpoint == v5.ENDPOINTS[0] else []
        body = canonical_json({"items": items, "totalCount": len(items)})
        return _Response(200, body, {"content-type": "application/json"})


class _Client:
    base_url = v5.BASE_URL
    api_key = "fixture-secret"

    def __init__(self, transport: _Transport) -> None:
        self.client = transport
        self.config = type("FixtureConfig", (), {"team_id": "team-fixture"})()


def _key(tmp_path: Path) -> tuple[Path, lifecycle.SigningIdentity]:
    key = tmp_path / "key"
    subprocess.run(
        [
            str(v5.OPENSSH_EXECUTABLE_PATH),
            "-q",
            "-t",
            "rsa",
            "-b",
            "2048",
            "-N",
            "",
            "-f",
            str(key),
        ],
        check=True,
        capture_output=True,
    )
    key_type, public, *_rest = key.with_suffix(".pub").read_text(encoding="ascii").split()
    allowed = f"mihir {key_type} {public}\n".encode()
    return key, lifecycle.SigningIdentity(
        principal="mihir",
        key_type=key_type,
        public_key_base64=public,
        fingerprint_sha256=v5._fingerprint(key_type, public),
        allowed_signers_sha256=sha256_bytes(allowed),
    )


def _billing_row(
    identifier: str = "billing-fixture",
    *,
    amount: float = 0.5,
    resource_type: str = "pod",
    resource_id: str | None = "pod-fixture",
) -> dict[str, object]:
    return {
        "id": identifier,
        "created_at": "1970-01-01T00:16:40+00:00",
        "updated_at": "1970-01-01T00:16:40+00:00",
        "last_billed_at": "1970-01-01T00:16:40+00:00",
        "amount_usd": amount,
        "currency": "USD",
        "resource_type": resource_type,
        "resource_id": resource_id,
    }


def _wallet(balance: float, *, billed: bool, resource_id: str | None = "pod-fixture") -> bytes:
    rows = [_billing_row(resource_id=resource_id)] if billed else []
    return _wallet_window(balance, len(rows), rows)


def _wallet_window(balance: float, total: int, rows: list[dict[str, object]]) -> bytes:
    return canonical_json(
        {
            "wallet_id": "wallet-fixture",
            "team_id": "team-fixture",
            "balance_usd": balance,
            "currency": "USD",
            "total_billings": total,
            "recent_billings": rows,
        }
    )

def _wallet_snapshot(raw: bytes, *, phase: str = "precreate") -> wallet_owner.WalletSnapshot:
    page = wallet_owner.parse_wallet_page(raw, maximum_total=4096)
    return wallet_owner.WalletSnapshot(
        wallet_id=page.wallet_id,
        team_id=page.team_id,
        currency=page.currency,
        balance=page.balance,
        total_billings=page.total_billings,
        rows=page.rows,
        evidence={"phase": phase, "fixture": True},
    )


def _paged_wallet(balance: float, rows: list[dict[str, object]]) -> Callable[[int, int], bytes]:
    def page(offset: int, _ordinal: int) -> bytes:
        return _wallet_window(balance, len(rows), rows[offset : offset + 100])
    return page

class _Commands:
    def __init__(
        self, tmp_path: Path, *, billing_delay: int = 0, remote_returncode: int = 0
    ) -> None:
        self.tmp_path = tmp_path
        self.pod_lists = 0
        self.terminated = False
        self.invocations: list[tuple[str, ...]] = []
        self.billing_delay = billing_delay
        self.wallet_after_polls = 0
        self.remote_returncode = remote_returncode
    def __call__(
        self, argv: Sequence[str], _input: bytes | None, _timeout: float
    ) -> lifecycle.CommandResult:
        args = tuple(argv)
        self.invocations.append(args)
        name = Path(args[0]).name.lower()
        output = b""
        if name == "ssh-keygen.exe":
            result = subprocess.run(args, capture_output=True, check=False, timeout=_timeout)
            return lifecycle.CommandResult(args, result.returncode, result.stdout, result.stderr)
        if "prime" in name:
            if "disks" in args:
                output = canonical_json({"disks": [], "total_count": 0, "offset": 0, "limit": 100})
            elif "status" in args:
                output = canonical_json(
                    {
                        "id": "pod-fixture",
                        "name": "ignored",
                        "status": "ACTIVE",
                        "sshConnection": "ubuntu@8.8.8.8 -p 2222",
                    }
                )
            elif "terminate" in args:
                self.terminated = True
            elif "pods" in args and "list" in args:
                self.pod_lists += 1
                visible = 2 <= self.pod_lists <= 3 and not self.terminated
                rows = (
                    [
                        {
                            "id": "pod-fixture",
                            "name": "redco-stage-d-prime-tests-once-v2-aaaaaaaaaaaa",
                        }
                    ]
                    if visible
                    else []
                )
                output = canonical_json(
                    {"pods": rows, "total_count": len(rows), "offset": 0, "limit": 100}
                )
        elif name == "ssh-keyscan.exe":
            output = (
                b"[8.8.8.8]:2222 ssh-rsa "
                b"AAAAB3NzaC1yc2EAAAADAQABAAABAQC3Gb/rOlR0rMY/ilVefyI9LXjn9n8XsO"
                b"pI/6wdeEDe8O7a2MweozgxrFFPuSx7H/W1WoR3apBwvVOOyyTK4YYMw1Yc/oSP"
                b"HlYg5nO1mEmZePV6YaQIGUk3UqkJXONYYdj7XBijrnmzI+w48DilRaQoOL98R1"
                b"4ClnIzi6V0eOibacw2u1RmjKLJl6FTAcUnjxdMQm+sl6/7xs5dhvSGg+06nZMu"
                b"v2ncyVDCKiibRxCykWMkUQTueryA/0/iiaTJ7ye5/Oz8/5WF5T2/J+xOj3bkxu"
                b"qkzAqsypLOVwp2UYNBkAS6LDMVq4ZFTIPlnj8zuvNgdkE43TRZXJPMI99FvTfB"
                b"\n"
            )
        elif name == "scp.exe" and any("gpu-facts.json" in item for item in args):
            destination = Path(args[-1])
            destination.joinpath("gpu-facts.json").write_bytes(
                canonical_json(
                    {
                        "schema_version": 2,
                        "device_count": 2,
                        "names": ["NVIDIA L40S", "NVIDIA L40S"],
                        "memory_bytes": [48_305_799_168, 48_305_799_168],
                        "selected_nominal_aggregate_gb": 96,
                        "observed_aggregate_bytes": 96_611_598_336,
                        "torch": "fixture",
                        "cuda": "fixture",
                    }
                )
            )
            cases = "".join(
                '<testcase classname="'
                + node.partition("::")[0].removesuffix(".py").replace("/", ".")
                + '" '
                f'name="{node.partition("::")[2]}"/>'
                for node in TEST_NODES
            )
            destination.joinpath("pytest.xml").write_text(
                f'<testsuite tests="5" failures="{int(bool(self.remote_returncode))}" '
                f'errors="0" skipped="0">{cases}</testsuite>',
                encoding="utf-8",
            )
            destination.joinpath("remote-status.json").write_bytes(
                canonical_json({"schema_version": 2, "returncode": self.remote_returncode})
            )
        returncode = self.remote_returncode if name == "ssh.exe" and "bash" in args else 0
        return lifecycle.CommandResult(args, returncode, output, b"")


def _context(
    tmp_path: Path, transport: _Transport, commands: _Commands
) -> lifecycle.RuntimeContext:
    key, identity = _key(tmp_path)
    uv = tmp_path / "uv"
    uv.write_bytes(b"fixture")
    transport.commands = commands
    return lifecycle.RuntimeContext(
        repository=tmp_path,
        authorization={
            "commit": "a" * 40,
            "tree": "b" * 40,
            "parent": "c" * 40,
            "authorization_path": AUTHORIZATION_PATH,
            "authorization_sha256": "d" * 64,
            "authorization_blob": "e" * 40,
        },
        client=_Client(transport),
        wallet_team_id="team-fixture",
        transport_errors=(OSError,),
        prime_executable=Path("prime.exe"),
        openssh={
            name: Path(cast(str, binding["path"])) for name, binding in OPENSSH_EXECUTABLES.items()
        },
        keygen_executable=v5.OPENSSH_EXECUTABLE_PATH,
        signing_key=key,
        identity=identity,
        linux_uv=uv,
        run=commands,
        now=lambda: 1_000,
        monotonic=lambda: 1.0,
        sleep=lambda _seconds: None,
    )

def _resign_terminal(
    context: lifecycle.RuntimeContext,
    root: Path,
    terminal: dict[str, Any],
    *,
    raw: bytes | None = None,
) -> None:
    terminal_raw = canonical_json(terminal) if raw is None else raw
    (root / "terminal.json").write_bytes(terminal_raw)
    signature = lifecycle._sign(context, terminal_raw, TERMINAL_NAMESPACE, timeout=30)
    (root / "terminal-envelope.json").write_bytes(
        evidence.signed_envelope(
            terminal_raw,
            signature,
            TERMINAL_NAMESPACE,
            context.identity,
            authority=READINESS_AUTHORITY,
        )
    )


def _replace_bound_bytes(
    root: Path, terminal: dict[str, Any], name: str, raw: bytes
) -> None:
    path = root / cast(str, cast(dict[str, Any], terminal["evidence_dag"])[name]["path"])
    path.write_bytes(raw)
    cast(dict[str, Any], terminal["evidence_dag"])[name] = {
        "path": evidence.ARTIFACT_FILENAMES[name],
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }

def _replace_bound_json(
    root: Path, terminal: dict[str, Any], name: str, value: object
) -> None:
    _replace_bound_bytes(root, terminal, name, canonical_json(value))


def _completed_with_history(tmp_path: Path) -> tuple[lifecycle.RuntimeContext, Path]:
    tmp_path.mkdir(parents=True)
    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    old_rows = [
        _billing_row(identifier="old-a", resource_id="historical-a"),
        _billing_row(identifier="old-b", resource_id="historical-b"),
    ]
    new_row = _billing_row(identifier="billing-fixture", resource_id="pod-fixture")
    def wallet_handler(offset: int, _ordinal: int) -> bytes:
        rows = [new_row, *old_rows] if commands.terminated else old_rows
        balance = 29.5 if commands.terminated else 30.0
        return _wallet_window(balance, len(rows), rows[offset : offset + 100])

    transport.wallet_handler = wallet_handler
    context = _context(tmp_path, transport, commands)
    assert lifecycle.run_one_shot(context).state == "completed"
    return context, tmp_path / EVIDENCE_ROOT


def _failed_fixture(tmp_path: Path) -> tuple[lifecycle.RuntimeContext, Path]:
    tmp_path.mkdir(parents=True)
    commands = _Commands(tmp_path, remote_returncode=1)
    transport = _Transport([_item(1)])
    old_rows = [_billing_row(identifier="old-a", resource_id="historical-a")]

    def wallet_handler(offset: int, _ordinal: int) -> bytes:
        if commands.terminated:
            raise OSError("fixture post-cleanup wallet failure")
        return _wallet_window(30.0, len(old_rows), old_rows[offset : offset + 100])

    transport.wallet_handler = wallet_handler
    context = _context(tmp_path, transport, commands)
    assert lifecycle.run_one_shot(context).state == "failed_terminal"
    return context, tmp_path / EVIDENCE_ROOT

def test_terminal_verifier_replays_wallet_projection_instead_of_trusting_digest(
    tmp_path: Path,
) -> None:
    context, root = _completed_with_history(tmp_path / "fixture")
    cleanup = json.loads((root / "cleanup.json").read_bytes())
    cleanup["wallet_after"]["support_cap_usd"] = "11.0"
    terminal = json.loads((root / "terminal.json").read_bytes())
    _replace_bound_json(root, terminal, "cleanup", cleanup)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError, match="financial law"):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize(
    "mutation",
    ["hash_only", "field_only", "coherent_invalid", "reorder", "remove", "duplicate"],
)
def test_resigned_terminal_rejects_every_coherent_wallet_row_mutation(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _completed_with_history(tmp_path / mutation)
    cleanup = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
    reconciliation = cast(dict[str, Any], cleanup["wallet_after"])
    after = cast(dict[str, Any], reconciliation["after_snapshot"])
    after_rows = cast(list[dict[str, Any]], after["rows"])
    new_rows = cast(list[dict[str, Any]], reconciliation["new_rows"])
    if mutation == "hash_only":
        after_rows[0]["semantic_row_sha256"] = "0" * 64
        new_rows[0]["semantic_row_sha256"] = "0" * 64
    elif mutation == "field_only":
        after_rows[0]["amount_usd"] = "0.4"
        new_rows[0]["amount_usd"] = "0.4"
    elif mutation == "coherent_invalid":
        for row in (after_rows[0], new_rows[0]):
            row["resource_id_sha256"] = sha256_bytes(b"unowned-pod")
            core = {key: value for key, value in row.items() if key != "semantic_row_sha256"}
            row["semantic_row_sha256"] = sha256_bytes(canonical_json(core))
    elif mutation == "reorder":
        after_rows[1], after_rows[2] = after_rows[2], after_rows[1]
    elif mutation == "remove":
        after_rows.pop()
        after["total_billings"] = 2
    else:
        after_rows.append(deepcopy(after_rows[-1]))
        after["total_billings"] = 4
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    _replace_bound_json(root, terminal, "cleanup", cleanup)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def test_resigned_completed_terminal_rejects_unknown_field_and_cleanup_error(
    tmp_path: Path,
) -> None:
    context, root = _completed_with_history(tmp_path / "combined")
    cleanup = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
    cleanup["errors"] = ["forged:semantic_failure"]
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    terminal["unknown_terminal_field"] = None
    _replace_bound_json(root, terminal, "cleanup", cleanup)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError, match="schema"):
        evidence.verify_terminal_evidence(root, context.identity)

    terminal.pop("unknown_terminal_field")
    terminal["cleanup_failures"] = ["forged:semantic_failure"]
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError, match="create or cleanup state"):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize("authority_name", sorted(READINESS_AUTHORITY))
def test_resigned_terminal_rejects_every_authority_escalation(
    tmp_path: Path, authority_name: str
) -> None:
    context, root = _completed_with_history(tmp_path / authority_name)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    cast(dict[str, object], terminal["authority"])[authority_name] = True
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError, match="authority"):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize(
    "mutation",
    [
        "unknown",
        "missing",
        "type",
        "state",
        "disposition",
        "purpose",
        "monitoring",
        "attempt",
        "retry",
        "assessment",
        "primary",
        "recovery",
        "create",
        "tests",
        "cleanup",
        "command_count",
        "prime_count",
        "wallet_count",
        "dag_unknown",
        "dag_binding_unknown",
        "cleanup_unknown",
        "cleanup_owned",
        "cleanup_terminated",
        "authorization_unknown",
    ],
)
def test_resigned_terminal_closed_schema_and_state_matrix(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    if mutation == "unknown":
        terminal["unknown"] = None
    elif mutation == "missing":
        terminal.pop("purpose")
    elif mutation == "type":
        terminal["tests_passed"] = 1
    elif mutation == "state":
        terminal["state"] = terminal["disposition"] = "no_qualifying_capacity"
    elif mutation == "disposition":
        terminal["disposition"] = "failed_terminal"
    elif mutation == "purpose":
        terminal["purpose"] = "model_training"
    elif mutation == "monitoring":
        terminal["monitoring"] = True
    elif mutation == "attempt":
        terminal["attempt_consumed"] = False
    elif mutation == "retry":
        terminal["retry"] = True
    elif mutation == "assessment":
        terminal["assessment_sha256"] = None
    elif mutation == "primary":
        terminal["primary_failure"] = "RuntimeError"
    elif mutation == "recovery":
        terminal["recovery_failures"] = ["RuntimeError"]
    elif mutation == "create":
        terminal["create_dispatched"] = False
    elif mutation == "tests":
        terminal["tests_passed"] = False
    elif mutation == "cleanup":
        terminal["cleanup_proven"] = False
    elif mutation == "command_count":
        terminal["command_count"] += 1
    elif mutation == "prime_count":
        terminal["prime_cli_call_count"] += 1
    elif mutation == "wallet_count":
        terminal["wallet_api_call_count"] += 1
    elif mutation == "dag_unknown":
        cast(dict[str, Any], terminal["evidence_dag"])["unknown"] = {
            "path": "unknown.json",
            "bytes": 0,
            "sha256": sha256_bytes(b""),
        }
    elif mutation == "dag_binding_unknown":
        cast(dict[str, Any], terminal["evidence_dag"])["cleanup"]["unknown"] = None
    elif mutation in {"cleanup_unknown", "cleanup_owned", "cleanup_terminated"}:
        cleanup = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
        if mutation == "cleanup_unknown":
            cleanup["unknown"] = None
        elif mutation == "cleanup_owned":
            cleanup["owned_identity_sha256s"] = []
        else:
            cleanup["terminated_identity_sha256s"] = []
        _replace_bound_json(root, terminal, "cleanup", cleanup)
    else:
        cast(dict[str, Any], terminal["authorization"])["unknown"] = None
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def test_signed_terminal_rejects_nonfinite_primitive_and_unknown_envelope_field(
    tmp_path: Path,
) -> None:
    context, root = _completed_with_history(tmp_path / "primitive")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    terminal["elapsed_seconds"] = float("nan")
    raw = json.dumps(terminal, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    _resign_terminal(context, root, terminal, raw=raw)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)

    context, root = _completed_with_history(tmp_path / "envelope")
    envelope = cast(dict[str, Any], json.loads((root / "terminal-envelope.json").read_bytes()))
    envelope["unknown"] = None
    (root / "terminal-envelope.json").write_bytes(canonical_json(envelope))
    with pytest.raises(ValueError, match="schema"):
        evidence.verify_terminal_evidence(root, context.identity)


def test_public_terminal_verifier_accepts_every_allowed_disposition(tmp_path: Path) -> None:
    cases: tuple[tuple[str, list[dict[str, object]], _Commands], ...] = (
        ("completed", [_item(1)], _Commands(tmp_path / "completed")),
        ("no_qualifying_capacity", [], _Commands(tmp_path / "none")),
        (
            "ambiguous_capacity",
            [_item(1), _item(2, cloudId="cloud-1", provider="other-provider")],
            _Commands(tmp_path / "ambiguous"),
        ),
        ("failed_terminal", [_item(1)], _Commands(tmp_path / "failed", remote_returncode=1)),
    )
    for expected, items, commands in cases:
        repository = commands.tmp_path
        repository.mkdir(parents=True)
        context = _context(repository, _Transport(items), commands)
        assert lifecycle.run_one_shot(context).state == expected
        verified = evidence.verify_terminal_evidence(repository / EVIDENCE_ROOT, context.identity)
        assert verified["state"] == verified["disposition"] == expected


@_parametrize("mutation", ["no_failure", "create_without_cleanup", "tests_without_remote"])
def test_failed_terminal_rejects_impossible_state_combinations(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _failed_fixture(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    if mutation == "no_failure":
        terminal["primary_failure"] = None
        terminal["recovery_failures"] = []
        terminal["cleanup_failures"] = []
        terminal["publication_failures"] = []
    elif mutation == "create_without_cleanup":
        (root / "cleanup.json").unlink()
        cast(dict[str, Any], terminal["evidence_dag"]).pop("cleanup")
        terminal["cleanup_proven"] = False
        terminal["cleanup_failures"] = []
    else:
        for name in ("gpu-facts", "junit", "remote-status"):
            (root / evidence.ARTIFACT_FILENAMES[name]).unlink()
            cast(dict[str, Any], terminal["evidence_dag"]).pop(name)
        terminal["tests_passed"] = True
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def _resign_assessment(
    context: lifecycle.RuntimeContext,
    root: Path,
    terminal: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    raw = canonical_json(assessment)
    _replace_bound_bytes(root, terminal, "assessment", raw)
    terminal["assessment_sha256"] = sha256_bytes(raw)
    signature = lifecycle._sign(context, raw, ASSESSMENT_NAMESPACE, timeout=30)
    envelope = evidence.signed_envelope(
        raw,
        signature,
        ASSESSMENT_NAMESPACE,
        context.identity,
    )
    _replace_bound_bytes(root, terminal, "assessment-envelope", envelope)


@_parametrize(
    "mutation",
    ["forged", "unknown", "missing", "type", "payload_hash", "qualifying_semantics"],
)
def test_resigned_terminal_rejects_transcript_schema_and_semantic_mutations(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    transcript = cast(dict[str, Any], json.loads((root / "transcript.json").read_bytes()))
    if mutation == "forged":
        transcript = {"forged": True}
    elif mutation == "unknown":
        transcript["unknown"] = None
    elif mutation == "missing":
        transcript.pop("request_count")
    elif mutation == "type":
        transcript["request_count"] = True
    elif mutation == "payload_hash":
        assessment = cast(dict[str, Any], json.loads((root / "assessment.json").read_bytes()))
        assessment["transcript_payload_sha256"] = "0" * 64
        _resign_assessment(context, root, terminal, assessment)
    else:
        page = cast(dict[str, Any], cast(list[object], transcript["pages"])[0])
        body = json.loads(base64.b64decode(page["decoded_application_body_b64"], validate=True))
        cast(dict[str, Any], cast(list[object], body["items"])[0])["isSpot"] = True
        body_raw = canonical_json(body)
        page["decoded_application_body_b64"] = base64.b64encode(body_raw).decode()
        page["decoded_application_body_bytes"] = len(body_raw)
        page["decoded_application_body_sha256"] = sha256_bytes(body_raw)
    if mutation != "payload_hash":
        _replace_bound_json(root, terminal, "transcript", transcript)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def _journal_records(root: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(raw))
        for raw in (root / "command-journal.jsonl").read_bytes().splitlines()
    ]


def _replace_journal(
    context: lifecycle.RuntimeContext, root: Path,
    terminal: dict[str, Any], records: list[dict[str, Any]],
) -> None:
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    _replace_bound_bytes(root, terminal, "command-journal", raw)
    _resign_terminal(context, root, terminal)


def _bind_journal(root: Path, terminal: dict[str, Any], records: list[dict[str, Any]]) -> None:
    for ordinal, record in enumerate(records, start=1):
        record["ordinal"] = ordinal
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    _replace_bound_bytes(root, terminal, "command-journal", raw)
    outcomes = [
        record["details"] for record in records if record["phase"] == "outcome"
        and type(record["details"]) is dict and "stdout_sha256" in record["details"]
    ]
    _replace_bound_json(root, terminal, "command-records", outcomes)


@_parametrize(
    "mutation",
    [
        "unknown_operation",
        "unknown_detail",
        "detail_type",
        "unpaired",
        "duplicate_outcome",
        "out_of_order",
        "wrong_operation_pair",
    ],
)
def test_resigned_terminal_rejects_journal_schema_and_pairing_mutations(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    records = _journal_records(root)
    if mutation == "unknown_operation":
        records[0]["operation"] = "unknown-operation"
    elif mutation == "unknown_detail":
        cast(dict[str, object], records[0]["details"])["unknown"] = None
    elif mutation == "detail_type":
        records[0]["details"] = []
    elif mutation == "unpaired":
        records.pop(1)
    elif mutation == "duplicate_outcome":
        records.insert(2, deepcopy(records[1]))
    elif mutation == "out_of_order":
        records[0], records[1] = records[1], records[0]
    else:
        records[1]["operation"] = "prime-api-create"
    for ordinal, record in enumerate(records, start=1):
        record["ordinal"] = ordinal
    _replace_journal(context, root, terminal, records)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize(
    "field,value",
    [
        ("raw_team_id", "team-secret"),
        ("provider_row", {"provider": "secret-provider"}),
        ("private_key_path", r"C:\\secret\\id_rsa"),
        ("credential", "fixture-secret"),
        ("raw_body", "provider-response"),
    ],
)
def test_resigned_terminal_rejects_every_journal_privacy_sentinel(
    tmp_path: Path, field: str, value: object
) -> None:
    context, root = _completed_with_history(tmp_path / field)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    records = _journal_records(root)
    cast(dict[str, object], records[0]["details"])[field] = value
    _replace_journal(context, root, terminal, records)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
    assert b"team-secret" not in canonical_json(terminal)


@_parametrize("artifact", ["create-dispatch", "create-result"])
def test_resigned_terminal_rejects_arbitrary_create_artifact(
    tmp_path: Path, artifact: str
) -> None:
    context, root = _completed_with_history(tmp_path / artifact)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    _replace_bound_json(root, terminal, artifact, {"forged": True})
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def test_resigned_terminal_rejects_arbitrary_handoff_payload(tmp_path: Path) -> None:
    context, root = _completed_with_history(tmp_path / "handoff")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    payload = canonical_json({"forged": True})
    signature = lifecycle._sign(context, payload, HANDOFF_NAMESPACE, timeout=30)
    envelope = evidence.signed_envelope(
        payload,
        signature,
        HANDOFF_NAMESPACE,
        context.identity,
        authority=READINESS_AUTHORITY,
    )
    _replace_bound_bytes(root, terminal, "handoff", payload)
    _replace_bound_bytes(root, terminal, "handoff-signature", signature)
    _replace_bound_bytes(root, terminal, "handoff-envelope", envelope)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize(
    "mutation",
    [
        "unknown", "missing", "authorization", "claim", "transcript", "assessment",
        "resource", "pod", "ssh", "runtime", "ttl", "retry", "authority", "path",
        "nonce", "signature_file", "signature_replay", "known_hash", "known_empty",
        "known_malformed", "known_oversize", "known_host", "known_port",
        "known_same_endpoint_key", "keyscan_digest", "keyscan_reorder",
        "keyscan_duplicate", "keyscan_missing",
    ],
)
def test_resigned_terminal_rejects_handoff_semantic_and_signature_matrix(
    tmp_path: Path, mutation: str
) -> None:
    context, root = _completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    handoff = cast(dict[str, Any], json.loads((root / "handoff.json").read_bytes()))
    if mutation == "unknown":
        handoff["unknown"] = None
    elif mutation == "missing":
        handoff.pop("state")
    elif mutation == "authorization":
        handoff["authorization"]["commit"] = "0" * 40
    elif mutation in {"claim", "transcript"}:
        handoff[mutation]["sha256"] = "0" * 64
    elif mutation == "assessment":
        handoff["assessment"]["envelope_sha256"] = "0" * 64
    elif mutation == "resource":
        handoff["selected_resource_sha256"] = "0" * 64
    elif mutation == "pod":
        handoff["pod"]["identity_sha256"] = "0" * 64
    elif mutation == "ssh":
        handoff["ssh"]["port"] = 0
    elif mutation == "runtime":
        handoff["runtime"]["test_nodes"].reverse()
    elif mutation == "ttl":
        handoff["expires_at_epoch"] += 1
    elif mutation == "retry":
        handoff["retry"] = True
    elif mutation == "authority":
        handoff["authority"]["science_authorized"] = True
    elif mutation == "path":
        handoff["evidence_paths"]["claim"] = "elsewhere.json"
    elif mutation == "nonce":
        handoff["nonce"] = "0"
    known_hosts = (root / "known-hosts.txt").read_bytes()
    known_mutations = {
        "known_empty": b"",
        "known_malformed": b"[8.8.8.8]:2222 ssh-rsa fixture\n",
        "known_oversize": b"x" * (64 * 1024 + 1),
        "known_host": known_hosts.replace(b"[8.8.8.8]:2222", b"[1.1.1.1]:2222"),
        "known_port": known_hosts.replace(b":2222", b":22"),
        "known_same_endpoint_key": known_hosts.replace(b"QC3G", b"QC4G", 1),
    }
    if mutation == "known_hash":
        handoff["ssh"]["known_hosts_sha256"] = "0" * 64
    elif mutation in known_mutations:
        known_hosts = known_mutations[mutation]
        handoff["ssh"]["known_hosts_sha256"] = sha256_bytes(known_hosts)
        _replace_bound_bytes(root, terminal, "known-hosts", known_hosts)
    if mutation.startswith("keyscan_"):
        records = _journal_records(root)
        scans = [
            index for index, record in enumerate(records) if record["phase"] == "dispatch"
            and cast(str, record["operation"]).startswith("ssh-keyscan.exe ")
        ]
        if mutation == "keyscan_digest":
            cast(dict[str, object], records[scans[0] + 1]["details"])["stdout_sha256"] = "0" * 64
        elif mutation == "keyscan_reorder":
            second = scans[1]
            records[second : second + 4] = (
                records[second + 2 : second + 4] + records[second : second + 2]
            )
        elif mutation == "keyscan_duplicate":
            records[scans[1] : scans[1]] = deepcopy(records[scans[0] : scans[0] + 2])
        else:
            del records[scans[1] : scans[1] + 2]
        _bind_journal(root, terminal, records)
    raw = canonical_json(handoff)
    envelope_signature = lifecycle._sign(
        context, b"replay" if mutation == "signature_replay" else raw,
        HANDOFF_NAMESPACE, timeout=30,
    )
    envelope = evidence.signed_envelope(
        raw, envelope_signature, HANDOFF_NAMESPACE, context.identity,
        authority=READINESS_AUTHORITY,
    )
    _replace_bound_bytes(root, terminal, "handoff", raw)
    signature_file = b"forged" if mutation == "signature_file" else envelope_signature
    _replace_bound_bytes(root, terminal, "handoff-signature", signature_file)
    _replace_bound_bytes(root, terminal, "handoff-envelope", envelope)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


@_parametrize("mutation", ["forged", "noncanonical", "identity", "pagination", "row", "digest"])
def test_resigned_terminal_rejects_wallet_before_without_after_matrix(
    tmp_path: Path, mutation: str,
) -> None:
    context, root = _failed_fixture(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    cleanup = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
    cleanup["wallet_after"] = None
    cleanup["errors"] = ["wallet:RuntimeError"]
    terminal["cleanup_proven"] = False
    terminal["cleanup_failures"] = ["wallet:RuntimeError"]
    wallet = cast(dict[str, Any], json.loads((root / "wallet-before.json").read_bytes()))
    if mutation == "forged":
        wallet = {"forged": True}
    elif mutation == "identity":
        wallet["wallet_identity_sha256"] = "0" * 64
    elif mutation == "pagination":
        wallet["pagination"]["pages"][0]["offset"] = 100
    elif mutation == "row":
        wallet["rows"][0]["amount_usd"] = "9"
    elif mutation == "digest":
        wallet["rows"][0]["semantic_row_sha256"] = "0" * 64
    wallet_raw = (
        json.dumps(wallet, indent=2).encode()
        if mutation == "noncanonical"
        else canonical_json(wallet)
    )
    _replace_bound_bytes(root, terminal, "wallet-before", wallet_raw)
    _replace_bound_json(root, terminal, "cleanup", cleanup)
    _resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
